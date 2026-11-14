"""RadImageNet backbone features: the MedMLX operation, its reusable model and declarations.

Image loading and standardization follow the upstream RadImageNet ACL example (a Keras
``ImageDataGenerator`` with Caffe-mode ``preprocess_input`` and ``rescale=1./255``).
Default outputs are a globally average-pooled vector and a secondary spatial map;
pooled-only jobs omit spatial/classifier transfer and artifacts. Four Keras backbone
families are supported, one MedMLX model ID each: ResNet50, DenseNet121, InceptionV3
and InceptionResNetV2. The checkpoint's detected architecture must match the model ID.
Supported saved models may also include a Dense classifier.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.files import PathLike, input_file, output_targets, publish
from medmlx_core.outputs import Artifact, artifact
from medmlx_core.weights import model_folder, weights_revision
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


def _declaration(model_id: str, label: str, peak_memory_gb: float) -> dict[str, str | float]:
    """Build the MedMLX registration for one backbone; all share ``extract``.

    ``peak_memory_gb`` is the measured MLX peak for a 1929x2207 chest radiograph.
    MedMLX reports it next to available memory but never refuses a run on it.
    """
    return {
        "id": model_id, "label": label, "task": "feature-extraction",
        "run": "radimagenet_mlx.medmlx:extract",
        "cohort": "radimagenet_mlx.medmlx:load_model",
        "summary": f"Radiology-pretrained {label} features for one 2D image.",
        "peak_memory_gb": peak_memory_gb,
    }


RESNET50 = _declaration("radimagenet-resnet50", "RadImageNet ResNet50", 0.2)
DENSENET121 = _declaration("radimagenet-densenet121", "RadImageNet DenseNet121", 0.2)
INCEPTIONV3 = _declaration("radimagenet-inceptionv3", "RadImageNet InceptionV3", 0.2)
INCEPTIONRESNETV2 = _declaration(
    "radimagenet-inceptionresnetv2", "RadImageNet InceptionResNetV2", 0.5,
)

type ModelId = Literal[
    "radimagenet-resnet50", "radimagenet-densenet121", "radimagenet-inceptionv3",
    "radimagenet-inceptionresnetv2",
]
ARCHITECTURES: dict[str, Architecture] = {
    "radimagenet-resnet50": Architecture.RESNET50,
    "radimagenet-densenet121": Architecture.DENSENET121,
    "radimagenet-inceptionv3": Architecture.INCEPTIONV3,
    "radimagenet-inceptionresnetv2": Architecture.INCEPTIONRESNETV2,
}


# Default (height, width) of ``prepare_image``. ``extract`` uses the checkpoint's fixed
# input size, falling back to 224 x 224 (ResNet50, DenseNet121) or 299 x 299 (Inception).
INPUT_SIZE = (224, 224)
# Every role is reserved on each run; roles a run does not write are removed.
OUTPUT_FILES = {
    "features": "features.npy",
    "spatial_features": "spatial_features.npy",
    "predictions": "predictions.npy",
    "metadata": "features.json",
}

type OutputSelection = Literal["all", "pooled"]

# The published file for each backbone inside a weights folder.
CHECKPOINT_FILES: dict[Architecture, str] = {
    Architecture.RESNET50: "RadImageNet-ResNet50_notop.safetensors",
    Architecture.DENSENET121: "RadImageNet-DenseNet121_notop.safetensors",
    Architecture.INCEPTIONV3: "RadImageNet-InceptionV3_notop.safetensors",
    Architecture.INCEPTIONRESNETV2: "RadImageNet-IRV2_notop.safetensors",
}


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


def _architecture(model: str) -> Architecture:
    """Return the model ID's backbone, rejecting unknown IDs before any work."""
    architecture = ARCHITECTURES.get(model)
    if architecture is None:
        raise InvalidInputError(f"model must be one of {', '.join(ARCHITECTURES)}: {model!r}")
    return architecture


def _selection(value: str) -> OutputSelection:
    """Admit the full artifact set or pooled features only."""
    if value not in ("all", "pooled"):
        raise InvalidInputError("output_selection must be 'all' or 'pooled'")
    return value


def _checkpoint(model: str, architecture: Architecture, weights_path: PathLike | None) -> Path:
    """Return the backbone's file in its weights folder, downloading the folder if omitted."""
    path = model_folder(model, weights_path) / CHECKPOINT_FILES[architecture]
    if not path.is_file():
        raise AssetNotReadyError(
            f"RadImageNet weights folder must contain {path.name}: {path.parent}",
            reason=f"missing {path.name}",
            hint="Run medmlx setup, or pass a weights_path folder that contains it.",
        )
    return path


def _path_ids(path: Path) -> set[Path | tuple[int, int]]:
    """Identify resolved paths and existing hard links during batch preflight."""
    result: set[Path | tuple[int, int]] = {path.resolve()}
    if path.exists():
        stat = path.stat()
        result.add((stat.st_dev, stat.st_ino))
    return result


def _batch_targets(
    image_paths: Sequence[PathLike], output_dirs: Sequence[PathLike], checkpoint: Path,
    overwrite: bool,
) -> tuple[tuple[Path, dict[str, Path]], ...]:
    """Preflight all batch paths in linear space/time before any output is replaced."""
    if not image_paths or len(image_paths) != len(output_dirs):
        raise InvalidInputError("image_paths and output_dirs must have the same nonzero length")
    sources = tuple(input_file("image_paths", value) for value in image_paths)
    inputs = _path_ids(checkpoint)
    for source in sources:
        inputs.update(_path_ids(source))
    jobs: list[tuple[Path, dict[str, Path]]] = []
    seen: set[Path | tuple[int, int]] = set()
    for source, output_dir in zip(sources, output_dirs, strict=True):
        targets = output_targets(
            output_dir, OUTPUT_FILES, overwrite=overwrite, inputs=[source, checkpoint]
        )
        for path in targets.values():
            identities = _path_ids(path)
            if identities & inputs:
                raise InvalidInputError("RadImageNet outputs must not alias any batch input")
            if identities & seen:
                raise InvalidInputError("RadImageNet batch outputs must not alias each other")
            seen.update(identities)
        jobs.append((source, targets))
    return tuple(jobs)


class RadImageNetModel:
    """One RadImageNet backbone reused for serial images or batches until closed.

    The weights folder is resolved (downloading it if omitted) when the model is
    created; the network loads after the first image decodes. A checkpoint file
    changed after that is refused. :meth:`extract` has the :func:`extract` contract
    without ``model`` and ``weights_path``; :meth:`extract_batch` bounds decoded
    images and native activations by ``batch_size``.
    """

    def __init__(self, *, model: ModelId, weights_path: PathLike | None = None) -> None:
        """Resolve the backbone's checkpoint without loading native weights."""
        self._architecture = _architecture(model)
        self.checkpoint = _checkpoint(model, self._architecture, weights_path)
        self._encoder: RadImageNetMlx | None = None
        self._closed = False
        stat = self.checkpoint.stat()
        self._identity = (
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino,
        )

    def __enter__(self) -> Self:
        """Return the model; a closed model raises ``RuntimeError``."""
        if self._closed:
            raise RuntimeError("RadImageNet model is closed")
        return self

    def __exit__(self, *_exception: object) -> None:
        """Close the model."""
        self.close()

    def close(self) -> None:
        """Release the network; later calls raise ``RuntimeError``."""
        self._encoder = None
        self._closed = True

    def _network(self) -> RadImageNetMlx:
        """Load once after input decode, rejecting changes to the bound file."""
        if self._closed:
            raise RuntimeError("RadImageNet model is closed")
        stat = self.checkpoint.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)
        if identity != self._identity:
            raise AssetNotReadyError("RadImageNet checkpoint changed; load the model again")
        if self._encoder is None:
            from radimagenet_mlx.mlx_model import RadImageNetMlx

            self._encoder = RadImageNetMlx(self.checkpoint, architecture=self._architecture)
        return self._encoder

    def extract(
        self, *, image_path: PathLike, output_dir: PathLike,
        output_selection: OutputSelection = "all", overwrite: bool = False,
    ) -> RunResult:
        """Write one image's features with the loaded network; see :func:`extract`."""
        selected = _selection(output_selection)
        source = input_file("image_path", image_path)
        targets = output_targets(
            output_dir, OUTPUT_FILES, overwrite=overwrite, inputs=[source, self.checkpoint]
        )
        with _decode_image(source) as decoded:
            return _extract_image(self._network(), self.checkpoint, decoded, targets, selected)

    def extract_batch(
        self, *, image_paths: Sequence[PathLike], output_dirs: Sequence[PathLike],
        batch_size: int = 8, output_selection: OutputSelection = "all", overwrite: bool = False,
    ) -> Iterator[RunResult]:
        """Stream model batches, yielding one :func:`extract` result per image in order.

        ``output_dirs`` gives one distinct folder per image; every path is checked
        before the first output is replaced. Fixed-batch checkpoints use their
        declared batch size, which must fit ``batch_size`` and divide the image count.
        Each image's files are published together; a later failure keeps the
        images already completed. Exhaust or close the iterator before closing the model.
        """
        selected = _selection(output_selection)
        if (
            isinstance(batch_size, bool)
            or not isinstance(cast(object, batch_size), int)
            or batch_size < 1
        ):
            raise InvalidInputError("batch_size must be a positive integer")
        jobs = _batch_targets(image_paths, output_dirs, self.checkpoint, overwrite)
        first: Image.Image | None = _decode_image(jobs[0][0])
        try:
            encoder = self._network()
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
                for index, (source, _targets) in enumerate(chunk):
                    if first is not None:
                        array[index] = _prepare_rgb(first, encoder.graph.input_size)[0]
                        first.close()
                        first = None
                    else:
                        with _decode_image(source) as decoded:
                            array[index] = _prepare_rgb(decoded, encoder.graph.input_size)[0]
                features, spatial, predictions = _infer(encoder, array, selected)
                for index, (_source, targets) in enumerate(chunk):
                    yield _write_features(
                        encoder, self.checkpoint, targets, features[index : index + 1],
                        None if spatial is None else spatial[index : index + 1],
                        None if predictions is None else predictions[index : index + 1],
                    )
                del features, spatial, predictions, array
        finally:
            if first is not None:
                first.close()


def load_model(*, model: ModelId, weights_path: PathLike | None = None) -> RadImageNetModel:
    """Load one RadImageNet backbone for several images, or for a MedMLX cohort.

    ``model`` and ``weights_path`` have the meanings they have in :func:`extract`.
    Close the returned model, or use it as a context manager, to release it.
    """
    return RadImageNetModel(model=model, weights_path=weights_path)


def _infer(
    encoder: RadImageNetMlx, array: FloatArray, selected: str,
) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
    """Select the pooled or full native graph boundary before host transfer."""
    if selected == "pooled":
        return encoder.features(array), None, None
    result: Inference = encoder.extract(array)
    return result.features, result.spatial, result.predictions


def extract(
    *,
    model: ModelId,
    image_path: PathLike,
    output_dir: PathLike,
    weights_path: PathLike | None = None,
    overwrite: bool = False,
    output_selection: OutputSelection = "all",
) -> RunResult:
    """Extract one 2D image's RadImageNet backbone features and any saved classifier output.

    Fixed checkpoint dimensions govern resizing, otherwise 224 for ResNet and DenseNet
    or 299 for Inception families, with the upstream preprocessing described in
    ``prepare_image``. Outputs carry schema version 3. Every argument, the image decode
    and the output paths are checked before weights download.

    Args:
        model: ``radimagenet-resnet50``, ``radimagenet-densenet121``,
            ``radimagenet-inceptionv3`` or ``radimagenet-inceptionresnetv2``.
        image_path: Pillow-readable image; the first frame is used.
        output_dir: Folder for the outputs; created if missing.
        weights_path: Weights folder holding the backbone's safetensors file (graph
            metadata and float32 weights), such as ``RadImageNet-ResNet50_notop.safetensors``.
            ``None`` downloads MedMLX/radimagenet-mlx.
        overwrite: Replace existing outputs; files a run does not write (spatial
            features, predictions) are then removed.
        output_selection: ``"all"`` (default) writes spatial features and any classifier
            output; ``"pooled"`` writes only pooled features and metadata.

    Returns:
        ``{"outputs": {...}, "artifacts": {...}, "details": {...}}`` with roles
        ``features`` (pooled ``(1, C)`` float32), ``metadata`` (``features.json``),
        ``spatial_features`` (``(1, H, W, C)``) unless pooled, and ``predictions``
        when the checkpoint has a classifier. ``details`` is the metadata.

    Raises:
        InvalidInputError: Unknown ``model``, missing or unreadable image, or invalid
            or existing output paths.
        AssetNotReadyError: Missing weights folder or file, invalid checkpoint, or its
            architecture is not the model's backbone.
    """
    architecture = _architecture(model)
    selected = _selection(output_selection)
    source = input_file("image_path", image_path)
    output_targets(output_dir, OUTPUT_FILES, overwrite=overwrite, inputs=[source])
    # Decode first so an unreadable image fails before any download.
    with _decode_image(source) as decoded:
        checkpoint = _checkpoint(model, architecture, weights_path)
        targets = output_targets(
            output_dir, OUTPUT_FILES, overwrite=overwrite, inputs=[source, checkpoint]
        )
        from radimagenet_mlx.mlx_model import RadImageNetMlx

        encoder = RadImageNetMlx(checkpoint, architecture=architecture)
        return _extract_image(encoder, checkpoint, decoded, targets, selected)


def _extract_image(
    encoder: RadImageNetMlx, checkpoint: Path, decoded: Image.Image, targets: dict[str, Path],
    selected: str,
) -> RunResult:
    """Prepare one decoded image, run the selected graph boundary and publish its files."""
    array = _prepare_rgb(decoded, encoder.graph.input_size)
    features, spatial, predictions = _infer(encoder, array, selected)
    return _write_features(encoder, checkpoint, targets, features, spatial, predictions)


def _write_features(
    encoder: RadImageNetMlx, checkpoint: Path, targets: dict[str, Path],
    features: FloatArray, spatial_features: FloatArray | None, predictions: FloatArray | None,
) -> RunResult:
    """Publish one image's selected arrays and metadata together."""
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
        targets["features"]: lambda path: np.save(path, features),
    }
    if spatial_features is not None:
        writes[targets["spatial_features"]] = lambda path: np.save(path, spatial_features)
        metadata["spatial_shape"] = list(spatial_features.shape)
    else:
        metadata["output_selection"] = "pooled"
    if predictions is not None:
        writes[targets["predictions"]] = lambda path: np.save(path, predictions)
        metadata["classifier_output"] = encoder.graph.output
        metadata["classification_shape"] = list(predictions.shape)
    writes[targets["metadata"]] = lambda path: path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    publish(writes, remove=[path for path in targets.values() if path not in writes])
    outputs: FeatureOutputs = {
        "features": targets["features"].as_posix(),
        "metadata": targets["metadata"].as_posix(),
    }
    tensor = artifact(kind="feature_tensor", coordinate_space="not_applicable")
    artifacts: dict[str, Artifact] = {
        "features": tensor,
        "metadata": artifact(kind="report", coordinate_space="not_applicable"),
    }
    if spatial_features is not None:
        outputs["spatial_features"] = targets["spatial_features"].as_posix()
        artifacts["spatial_features"] = tensor
    if predictions is not None:
        outputs["predictions"] = targets["predictions"].as_posix()
        artifacts["predictions"] = tensor
    return {"outputs": outputs, "artifacts": artifacts, "details": metadata}
