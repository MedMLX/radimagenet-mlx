"""MedMLX declaration and end-to-end runner for RadImageNet backbone features.

Image loading and standardization follow the upstream RadImageNet ACL example (a Keras
``ImageDataGenerator`` with Caffe-mode ``preprocess_input`` and ``rescale=1./255``).
Default outputs are a globally average-pooled vector and a secondary spatial map;
pooled-only jobs omit spatial/classifier transfer and artifacts. Four
Keras backbone families are supported: ResNet50, DenseNet121, InceptionV3 and
InceptionResNetV2; the checkpoint's architecture is detected automatically, or
checked against an explicit ``backbone`` argument. Supported saved models may also
include a Dense classifier.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Self, cast

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.weights import weights_dir, weights_revision
from PIL import Image

from radimagenet_mlx._typing import (
    FeatureMetadata,
    FeatureOutputs,
    FloatArray,
    RunResult,
)
from radimagenet_mlx.features import POOLED_OUTPUT_LAYER
from radimagenet_mlx.graph import Architecture, Input

if TYPE_CHECKING:
    from radimagenet_mlx.mlx_model import Inference, RadImageNetMlx


def _declaration(
    model_id: str, label: str, runner: str, peak_memory_gb: float,
) -> dict[str, str | float]:
    """Build the MedMLX registration for one backbone runner.

    ``peak_memory_gb`` is the measured MLX peak for a 1929x2207 chest radiograph.
    MedMLX reports it next to available memory but never refuses a run on it.
    """
    return {
        "id": model_id, "label": label, "task": "feature-extraction",
        "run": f"radimagenet_mlx.medmlx:{runner}",
        "summary": f"Radiology-pretrained {label} features for one 2D image.",
        "peak_memory_gb": peak_memory_gb,
    }


RESNET50 = _declaration("radimagenet-resnet50", "RadImageNet ResNet50", "resnet50", 0.2)
DENSENET121 = _declaration(
    "radimagenet-densenet121", "RadImageNet DenseNet121", "densenet121", 0.2,
)
INCEPTIONV3 = _declaration("radimagenet-inceptionv3", "RadImageNet InceptionV3", "inceptionv3", 0.2)
INCEPTIONRESNETV2 = _declaration(
    "radimagenet-inceptionresnetv2", "RadImageNet InceptionResNetV2", "inceptionresnetv2", 0.5,
)


# Default (height, width) of ``prepare_image``. ``run`` uses the checkpoint's fixed input
# size, falling back to 224 x 224 (ResNet50, DenseNet121) or 299 x 299 (Inception).
INPUT_SIZE = (224, 224)
FEATURES_NPY = "features.npy"
SPATIAL_FEATURES_NPY = "spatial_features.npy"
FEATURES_JSON = "features.json"
PREDICTIONS_NPY = "predictions.npy"


def prepare_image(path: Path, size: tuple[int, int] = INPUT_SIZE) -> FloatArray:
    """Return prepared ``(1, H, W, 3)`` float32 data for ``size=(height, width)``.

    Converts to RGB (grayscale is replicated across channels, alpha is dropped) and
    resizes to ``size`` with nearest-neighbor interpolation, without preserving the
    aspect ratio. Then reverses RGB to BGR, subtracts the per-channel ImageNet means
    ``[103.939, 116.779, 123.68]`` (no division by a standard deviation) and
    multiplies by ``1./255``. This matches the upstream RadImageNet ACL example's
    ``ImageDataGenerator`` with ``preprocessing_function=imagenet_utils.preprocess_input``
    and ``rescale=1./255``: ``preprocess_input`` (default ``mode="caffe"``) runs before
    ``rescale``. One route serves all four backbones; only ``size`` differs.

    Raises:
        InvalidInputError: If Pillow cannot read ``path`` as a 2D image.
    """

    with _decode_image(path) as image:
        return _prepare_rgb(image, size)


def _decode_image(path: Path) -> Image.Image:
    """Fully decode the first frame into an owned RGB image before loading weights."""
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
            return image
    except (OSError, ValueError) as error:
        raise InvalidInputError(
            f"RadImageNet requires a 2D image file Pillow can read (PNG or JPEG): {path}: {error}"
        ) from error


def _prepare_rgb(image: Image.Image, size: tuple[int, int]) -> FloatArray:
    """Resize a decoded RGB image once, then apply the unchanged upstream transform."""
    with image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST) as resized:
        array = np.asarray(resized, dtype=np.float32)[None, ...]
    array = array[..., ::-1].copy()
    array -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    array *= 1.0 / 255
    return array


def _check_outputs(output: Path, overwrite: bool, inputs: tuple[Path, ...]) -> None:
    """Reject a non-bool ``overwrite``, an invalid ``output_dir``, existing outputs
    without overwrite, and outputs that alias an input or each other."""
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be true or false")
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {output}")
    names = (FEATURES_NPY, SPATIAL_FEATURES_NPY, FEATURES_JSON, PREDICTIONS_NPY)
    targets = tuple(output / name for name in names)
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise InvalidInputError(
            f"RadImageNet outputs already exist: {', '.join(existing)}; "
            "pass overwrite=true to replace them"
        )
    if any(_paths_alias(path, source) for path in targets for source in inputs):
        raise InvalidInputError("RadImageNet outputs must not alias inputs")
    if any(_paths_alias(first, second) for first, second in combinations(targets, 2)):
        raise InvalidInputError("RadImageNet outputs must not alias each other")


def _architecture(backbone: Architecture | str | None) -> Architecture | None:
    """Return the requested backbone, rejecting unknown names before any work."""
    if backbone is None:
        return None
    try:
        return Architecture(backbone)
    except ValueError as error:
        choices = ", ".join(architecture.value for architecture in Architecture)
        raise InvalidInputError(f"backbone must be one of {choices}; got {backbone!r}") from error


def _publish(writes: dict[Path, Callable[[Path], object]]) -> None:
    """Stage outputs in a private directory, then rename them into place.

    Earlier outputs stay untouched until every new file is complete, and a failed
    write leaves no partial files behind.
    """
    with TemporaryDirectory(prefix=".radimagenet-mlx-", dir=next(iter(writes)).parent) as staging:
        partials = {path: Path(staging) / path.name for path in writes}
        for path, write in writes.items():
            write(partials[path])
        for path, partial in partials.items():
            os.replace(partial, path)


def _paths_alias(first: Path, second: Path) -> bool:
    """Detect resolved-path or existing-file aliases between two paths."""
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def _selection(value: str) -> str:
    """Admit the full legacy artifact set or pooled features only."""
    if value not in ("all", "pooled"):
        raise InvalidInputError("output_selection must be 'all' or 'pooled'")
    return value


def _checkpoint_path(value: str) -> Path:
    """Check the model path without loading native weights."""
    path = Path(value).expanduser()
    if not path.is_file():
        raise AssetNotReadyError(
            f"Missing RadImageNet model: {path}", reason=f"missing {path}",
            hint="Pass weights_path pointing to a supported safetensors model file.",
        )
    return path


def _source_path(value: str) -> Path:
    """Require a local image path before decoding or loading a model."""
    path = Path(value).expanduser()
    if not path.is_file():
        raise InvalidInputError(f"Missing input image: {path}")
    return path


def _path_ids(path: Path) -> set[Path | tuple[int, int]]:
    """Identify resolved paths and existing hard links during cohort preflight."""
    result: set[Path | tuple[int, int]] = {path.resolve()}
    if path.exists():
        stat = path.stat()
        result.add((stat.st_dev, stat.st_ino))
    return result


def _batch_paths(
    image_paths: Sequence[str], output_dirs: Sequence[str], checkpoint: Path, overwrite: bool,
) -> tuple[tuple[Path, Path], ...]:
    """Preflight all cohort paths in linear space/time before any output is replaced."""
    if not image_paths or len(image_paths) != len(output_dirs):
        raise InvalidInputError("image_paths and output_dirs must have the same nonzero length")
    sources = tuple(_source_path(value) for value in image_paths)
    jobs = tuple(zip(sources, (Path(value).expanduser() for value in output_dirs), strict=True))
    inputs = _path_ids(checkpoint)
    for source in sources:
        inputs.update(_path_ids(source))
    targets: set[Path | tuple[int, int]] = set()
    for source, output in jobs:
        _check_outputs(output, overwrite, (source, checkpoint))
        for name in (FEATURES_NPY, SPATIAL_FEATURES_NPY, FEATURES_JSON, PREDICTIONS_NPY):
            identities = _path_ids(output / name)
            if identities & inputs:
                raise InvalidInputError("RadImageNet outputs must not alias any cohort input")
            if identities & targets:
                raise InvalidInputError("RadImageNet cohort outputs must not alias each other")
            targets.update(identities)
    return jobs


class FeatureSession:
    """Reuse one lazily loaded native model for sequential image or batch jobs.

    Use as a context manager to release model references when the cohort ends.
    Inputs are fully decoded before inference; the first image is decoded before
    the model loads. ``run_batch`` bounds decoded/prepared images and native
    activations by ``batch_size`` and yields the usual result per image.
    """

    def __init__(self, weights_path: str, *, backbone: Architecture | str | None = None):
        """Validate a checkpoint path and architecture without loading native weights."""
        self._required = _architecture(backbone)
        self.checkpoint = _checkpoint_path(weights_path)
        self._encoder: RadImageNetMlx | None = None
        self._closed = False
        stat = self.checkpoint.stat()
        self._identity = (
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino,
        )

    def __enter__(self) -> Self:
        """Return the session; a closed session raises ``InvalidInputError``."""
        if self._closed:
            raise InvalidInputError("FeatureSession is closed")
        return self

    def __exit__(self, *_exception: object) -> None:
        """Close the session."""
        self.close()

    def close(self) -> None:
        """Release this session's model and prevent further work."""
        self._encoder = None
        self._closed = True

    def _model(self) -> RadImageNetMlx:
        """Load once after input decode, rejecting changes to an already bound file."""
        if self._closed:
            raise InvalidInputError("FeatureSession is closed")
        stat = self.checkpoint.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)
        if identity != self._identity:
            raise AssetNotReadyError("RadImageNet checkpoint changed; create a new FeatureSession")
        if self._encoder is None:
            from radimagenet_mlx.mlx_model import RadImageNetMlx

            self._encoder = RadImageNetMlx(self.checkpoint, architecture=self._required)
        return self._encoder

    def run(
        self, *, image_path: str, output_dir: str, output_selection: str = "all",
        overwrite: bool = False, _decoded: Image.Image | None = None,
    ) -> RunResult:
        """Write one image's selected artifacts, preserving the default full output contract."""
        selected = _selection(output_selection)
        source, output = _source_path(image_path), Path(output_dir).expanduser()
        _check_outputs(output, overwrite, (source, self.checkpoint))
        if _decoded is None:
            with _decode_image(source) as decoded:
                return self._run_decoded(decoded, output, selected)
        return self._run_decoded(_decoded, output, selected)

    def _run_decoded(self, decoded: Image.Image, output: Path, selected: str) -> RunResult:
        encoder = self._model()
        array = _prepare_rgb(decoded, encoder.graph.input_size)
        features, spatial, predictions = _infer(encoder, array, selected)
        return _write_features(encoder, self.checkpoint, output, features, spatial, predictions)

    def run_batch(
        self, *, image_paths: Sequence[str], output_dirs: Sequence[str], batch_size: int = 8,
        output_selection: str = "all", overwrite: bool = False,
    ) -> Iterator[RunResult]:
        """Stream real model batches, yielding schema-3 artifacts per image in input order.

        ``output_dirs`` provides one distinct folder per input. Fixed-batch models
        use their declared batch size, which must fit the bound and divide the
        cohort length. Each completed image is published atomically; later failures
        retain earlier completed jobs. Exhaust or close the iterator before closing
        this session.
        """
        selected = _selection(output_selection)
        if (
            isinstance(batch_size, bool)
            or not isinstance(cast(object, batch_size), int)
            or batch_size < 1
        ):
            raise InvalidInputError("batch_size must be a positive integer")
        jobs = _batch_paths(image_paths, output_dirs, self.checkpoint, overwrite)
        first: Image.Image | None = _decode_image(jobs[0][0])
        try:
            encoder = self._model()
            spec = encoder.graph.nodes[0].operation
            if not isinstance(spec, Input):
                raise AssetNotReadyError("RadImageNet graph must start with an input")
            capacity = batch_size if spec.batch is None else spec.batch
            if capacity > batch_size or (len(jobs) % capacity and spec.batch is not None):
                raise InvalidInputError(
                    "fixed checkpoint batch must fit batch_size and divide image count"
                )
            for start in range(0, len(jobs), capacity):
                chunk = jobs[start : start + capacity]
                array = np.empty((len(chunk), *encoder.graph.input_size, 3), dtype=np.float32)
                for index, (source, _output) in enumerate(chunk):
                    if first is not None:
                        array[index] = _prepare_rgb(first, encoder.graph.input_size)[0]
                        first.close()
                        first = None
                    else:
                        with _decode_image(source) as decoded:
                            array[index] = _prepare_rgb(decoded, encoder.graph.input_size)[0]
                features, spatial, predictions = _infer(encoder, array, selected)
                for index, (_source, output) in enumerate(chunk):
                    yield _write_features(
                        encoder, self.checkpoint, output, features[index : index + 1],
                        None if spatial is None else spatial[index : index + 1],
                        None if predictions is None else predictions[index : index + 1],
                    )
                del features, spatial, predictions, array
        finally:
            if first is not None:
                first.close()


def _infer(
    encoder: RadImageNetMlx, array: FloatArray, selected: str,
) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
    """Select the existing pooled or full native graph boundary before host transfer."""
    if selected == "pooled":
        return encoder.features(array), None, None
    result: Inference = encoder.extract(array)
    return result.features, result.spatial, result.predictions


def run(
    *,
    image_path: str,
    weights_path: str,
    output_dir: str,
    backbone: Architecture | str | None = None,
    overwrite: bool = False,
    output_selection: str = "all",
) -> RunResult:
    """Extract native backbone features and any supplied classifier from a saved model.

    Detect all four official architectures; optionally require ``backbone``.
    Fixed checkpoint dimensions govern resizing, otherwise use 224 for ResNet
    and DenseNet or 299 for Inception families. All use the upstream preprocessing
    described in ``prepare_image``. Outputs carry schema version 3. In full-output
    mode ``predictions.npy`` exists only for a saved classifier. Existing outputs require explicit
    ``overwrite=True``. Output aliases of either input or another output are
    rejected before inference. These checks do not prevent concurrent path changes.

    Args:
        image_path: Path to a Pillow-readable image; the first frame is used.
        weights_path: Local safetensors file containing graph metadata and float32 weights.
            Weights are not downloaded.
        output_dir: Directory for pooled and spatial features and metadata;
            classifier predictions are also written when present in the model.
        backbone: Architecture to require: ``resnet50``, ``densenet121``,
            ``inceptionv3`` or ``inceptionresnetv2``. Defaults to ``None`` for
            automatic detection.
        overwrite: Allow existing output files to be replaced (default ``False``);
            a stale predictions file is removed when the selected model has no classifier.
            Pooled-only output removes earlier spatial and prediction files on success.
            New outputs replace earlier ones only after the whole run succeeds.
        output_selection: ``"all"`` (default) retains spatial features and any
            classifier output; ``"pooled"`` writes only pooled features and metadata.

    Returns:
        ``{"outputs": {...}, "details": {...}}`` with absolute POSIX output paths and
        the metadata also written to ``features.json``.

    Raises:
        InvalidInputError: Missing or unreadable image, unknown ``backbone``,
            ``overwrite`` not a bool, ``output_dir`` not a directory, existing outputs
            without ``overwrite``, or output aliases.
        AssetNotReadyError: Missing or invalid checkpoint, or its architecture is not
            the requested ``backbone``.
    """

    _source_path(image_path)
    _selection(output_selection)
    with FeatureSession(weights_path, backbone=backbone) as session:
        return session.run(
            image_path=image_path, output_dir=output_dir,
            overwrite=overwrite, output_selection=output_selection,
        )


def run_batch(
    *, image_paths: Sequence[str], weights_path: str, output_dirs: Sequence[str],
    batch_size: int = 8, backbone: Architecture | str | None = None,
    output_selection: str = "all", overwrite: bool = False,
) -> Iterator[RunResult]:
    """Load one model for a bounded cohort and yield the usual per-image results.

    Each image has its own ``output_dirs`` folder. Consume or close the iterator
    to release its model; use ``FeatureSession`` to reuse it across multiple cohorts.
    See ``FeatureSession.run_batch`` for fixed-batch and partial-completion behavior.
    """
    with FeatureSession(weights_path, backbone=backbone) as session:
        yield from session.run_batch(
            image_paths=image_paths, output_dirs=output_dirs, batch_size=batch_size,
            output_selection=output_selection, overwrite=overwrite,
        )


def _write_features(
    encoder: RadImageNetMlx, checkpoint: Path, output: Path,
    features: FloatArray, spatial_features: FloatArray | None, predictions: FloatArray | None,
) -> RunResult:
    """Publish one image's selected arrays while preserving the default schema and names."""
    features_npy, spatial_features_npy = output / FEATURES_NPY, output / SPATIAL_FEATURES_NPY
    features_json, predictions_npy = output / FEATURES_JSON, output / PREDICTIONS_NPY
    detected = encoder.graph.architecture
    metadata: FeatureMetadata = {
        "schema_version": 3,
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "model_path": checkpoint.absolute().as_posix(),
        "weights_revision": weights_revision(checkpoint.parent),
        "backbone": detected,
        "feature_layer": POOLED_OUTPUT_LAYER,
        "spatial_feature_layer": encoder.graph.spatial,
        "preprocessing": "upstream",
        "interpolation": "nearest",
        "backend": "mlx",
    }
    writes: dict[Path, Callable[[Path], object]] = {
        features_npy: lambda path: np.save(path, features),
    }
    if spatial_features is not None:
        writes[spatial_features_npy] = lambda path: np.save(path, spatial_features)
        metadata["spatial_shape"] = list(spatial_features.shape)
    else:
        metadata["output_selection"] = "pooled"
    if predictions is not None:
        writes[predictions_npy] = lambda path: np.save(path, predictions)
        metadata["classifier_output"] = encoder.graph.output
        metadata["classification_shape"] = list(predictions.shape)
    writes[features_json] = lambda path: path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    output.mkdir(parents=True, exist_ok=True)
    _publish(writes)
    outputs: FeatureOutputs = {
        "features_npy": features_npy.resolve().as_posix(),
        "features_json": features_json.resolve().as_posix(),
    }
    if spatial_features is not None:
        outputs["spatial_features_npy"] = spatial_features_npy.resolve().as_posix()
    elif spatial_features_npy.exists() or spatial_features_npy.is_symlink():
        spatial_features_npy.unlink()
    if predictions is not None:
        outputs["predictions_npy"] = predictions_npy.resolve().as_posix()
    elif predictions_npy.exists() or predictions_npy.is_symlink():
        # Only reachable with overwrite=True; remove the stale file once the run succeeded.
        predictions_npy.unlink()
    return {"outputs": outputs, "details": metadata}



def _run_backbone(
    model_id: str, filename: str, architecture: str, *, image_path: str,
    output_dir: str, weights_path: str | None, overwrite: bool,
    output_selection: str,
) -> RunResult:
    """Resolve the shared model folder and select the registered backbone's file."""
    source = _source_path(image_path)
    _selection(output_selection)
    _check_outputs(Path(output_dir).expanduser(), overwrite, (source,))
    with _decode_image(source) as decoded:
        directory = (Path(weights_path).expanduser()
                     if weights_path is not None else weights_dir(model_id))
        if not directory.is_dir():
            raise AssetNotReadyError(f"Missing RadImageNet weights folder: {directory}")
        checkpoint = directory / filename
        if not checkpoint.is_file():
            raise AssetNotReadyError(
                f"RadImageNet weights folder must contain {filename}: {directory}",
                hint=f"Run medmlx setup {model_id}, or pass a complete weights_path folder.",
            )
        with FeatureSession(str(checkpoint), backbone=architecture) as session:
            return session.run(
                image_path=str(source), output_dir=output_dir, overwrite=overwrite,
                output_selection=output_selection, _decoded=decoded,
            )


def resnet50(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
    output_selection: str = "all",
) -> RunResult:
    """Extract ResNet50 features; omitted weights download MedMLX/radimagenet-mlx.

    Args:
        image_path: Pillow-readable 2D image.
        output_dir: Directory for features and metadata; existing outputs require overwrite.
        weights_path: Folder containing RadImageNet-ResNet50_notop.safetensors; None downloads it.
        overwrite: Allow replacing existing outputs.
        output_selection: ``"all"`` (default) or ``"pooled"`` for features and metadata only.
    """
    return _run_backbone(
        "radimagenet-resnet50", "RadImageNet-ResNet50_notop.safetensors", "resnet50",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite, output_selection=output_selection,
    )


def densenet121(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
    output_selection: str = "all",
) -> RunResult:
    """Extract DenseNet121 features from an optional weights_path folder.

    Omit weights_path to download MedMLX/radimagenet-mlx. Explicit folders
    must contain RadImageNet-DenseNet121_notop.safetensors. Input/output behavior matches resnet50.
    """
    return _run_backbone(
        "radimagenet-densenet121", "RadImageNet-DenseNet121_notop.safetensors", "densenet121",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite, output_selection=output_selection,
    )


def inceptionv3(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
    output_selection: str = "all",
) -> RunResult:
    """Extract InceptionV3 features from an optional weights_path folder.

    Omit weights_path to download MedMLX/radimagenet-mlx. Explicit folders
    must contain RadImageNet-InceptionV3_notop.safetensors. Input/output behavior matches resnet50.
    """
    return _run_backbone(
        "radimagenet-inceptionv3", "RadImageNet-InceptionV3_notop.safetensors", "inceptionv3",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite, output_selection=output_selection,
    )


def inceptionresnetv2(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
    output_selection: str = "all",
) -> RunResult:
    """Extract InceptionResNetV2 features from an optional weights_path folder.

    Omit weights_path to download MedMLX/radimagenet-mlx. Explicit folders
    must contain RadImageNet-IRV2_notop.safetensors. Input/output behavior matches resnet50.
    """
    return _run_backbone(
        "radimagenet-inceptionresnetv2", "RadImageNet-IRV2_notop.safetensors", "inceptionresnetv2",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite, output_selection=output_selection,
    )
