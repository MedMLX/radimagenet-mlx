"""MedMLX declaration and end-to-end runner for RadImageNet backbone features.

Image loading and standardization follow the upstream RadImageNet ACL example (a Keras
``ImageDataGenerator`` with Caffe-mode ``preprocess_input`` and ``rescale=1./255``).
Outputs are a globally average-pooled vector and a secondary spatial map. Four
Keras backbone families are supported: ResNet50, DenseNet121, InceptionV3 and
InceptionResNetV2; the checkpoint's architecture is detected automatically, or
checked against an explicit ``backbone`` argument. Supported saved models may also
include a Dense classifier.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

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
from radimagenet_mlx.graph import Architecture


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

    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image = image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
    except (OSError, ValueError) as error:
        raise InvalidInputError(
            f"RadImageNet requires a 2D image file Pillow can read (PNG or JPEG): {path}: {error}"
        ) from error
    array = np.asarray(image, dtype=np.float32)[None, ...]
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


def run(
    *,
    image_path: str,
    weights_path: str,
    output_dir: str,
    backbone: Architecture | str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Extract native backbone features and any supplied classifier from a saved model.

    Detect all four official architectures; optionally require ``backbone``.
    Fixed checkpoint dimensions govern resizing, otherwise use 224 for ResNet
    and DenseNet or 299 for Inception families. All use the upstream preprocessing
    described in ``prepare_image``. Outputs carry schema version 3. ``predictions.npy``
    exists only for a saved classifier. Existing outputs require explicit
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
            New outputs replace earlier ones only after the whole run succeeds.

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

    source = Path(image_path)
    if not source.is_file():
        raise InvalidInputError(f"Missing input image: {source}")
    required = _architecture(backbone)
    checkpoint = Path(weights_path)
    if not checkpoint.is_file():
        raise AssetNotReadyError(
            f"Missing RadImageNet model: {checkpoint}",
            reason=f"missing {checkpoint}",
            hint="Pass weights_path pointing to a supported safetensors model file.",
        )
    output = Path(output_dir)
    _check_outputs(output, overwrite, (source, checkpoint))
    features_npy, spatial_features_npy = output / FEATURES_NPY, output / SPATIAL_FEATURES_NPY
    features_json, predictions_npy = output / FEATURES_JSON, output / PREDICTIONS_NPY

    # Read the image before loading weights so input errors remain actionable.
    prepare_image(source)
    from radimagenet_mlx.mlx_model import RadImageNetMlx

    encoder = RadImageNetMlx(checkpoint)
    detected = encoder.graph.architecture
    if required is not None and required != detected:
        raise AssetNotReadyError(
            f"RadImageNet checkpoint architecture is {detected!r}, not requested {backbone!r}"
        )
    array = prepare_image(source, encoder.graph.input_size)
    inference = encoder.extract(array)
    spatial_features, features = inference.spatial, inference.features
    predictions = inference.predictions
    metadata: FeatureMetadata = {
        "schema_version": 3,
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "model_path": checkpoint.absolute().as_posix(),
        "weights_revision": weights_revision(checkpoint.parent),
        "backbone": detected,
        "feature_layer": POOLED_OUTPUT_LAYER,
        "spatial_feature_layer": encoder.graph.spatial,
        "spatial_shape": list(spatial_features.shape),
        "preprocessing": "upstream",
        "interpolation": "nearest",
        "backend": "mlx",
    }
    writes: dict[Path, Callable[[Path], object]] = {
        features_npy: lambda path: np.save(path, features),
        spatial_features_npy: lambda path: np.save(path, spatial_features),
    }
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
        "spatial_features_npy": spatial_features_npy.resolve().as_posix(),
        "features_json": features_json.resolve().as_posix(),
    }
    if predictions is not None:
        outputs["predictions_npy"] = predictions_npy.resolve().as_posix()
    elif predictions_npy.exists() or predictions_npy.is_symlink():
        # Only reachable with overwrite=True; remove the stale file once the run succeeded.
        predictions_npy.unlink()
    return {"outputs": outputs, "details": metadata}



def _run_backbone(
    model_id: str, filename: str, architecture: str, *, image_path: str,
    output_dir: str, weights_path: str | None, overwrite: bool,
) -> RunResult:
    """Resolve the shared model folder and select the registered backbone's file."""
    source = Path(image_path).expanduser()
    if not source.is_file():
        raise InvalidInputError(f"Missing input image: {source}")
    _check_outputs(Path(output_dir).expanduser(), overwrite, (source,))
    prepare_image(source)
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
    return run(
        image_path=str(source), weights_path=str(checkpoint),
        output_dir=str(Path(output_dir).expanduser()), backbone=architecture, overwrite=overwrite,
    )


def resnet50(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Extract ResNet50 features; omitted weights download MedMLX/radimagenet-mlx.

    Args:
        image_path: Pillow-readable 2D image.
        output_dir: Directory for features and metadata; existing outputs require overwrite.
        weights_path: Folder containing RadImageNet-ResNet50_notop.safetensors; None downloads it.
        overwrite: Allow replacing existing outputs.
    """
    return _run_backbone(
        "radimagenet-resnet50", "RadImageNet-ResNet50_notop.safetensors", "resnet50",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite,
    )


def densenet121(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Extract DenseNet121 features from an optional weights_path folder.

    Omit weights_path to download MedMLX/radimagenet-mlx. Explicit folders
    must contain RadImageNet-DenseNet121_notop.safetensors. Input/output behavior matches resnet50.
    """
    return _run_backbone(
        "radimagenet-densenet121", "RadImageNet-DenseNet121_notop.safetensors", "densenet121",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite,
    )


def inceptionv3(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Extract InceptionV3 features from an optional weights_path folder.

    Omit weights_path to download MedMLX/radimagenet-mlx. Explicit folders
    must contain RadImageNet-InceptionV3_notop.safetensors. Input/output behavior matches resnet50.
    """
    return _run_backbone(
        "radimagenet-inceptionv3", "RadImageNet-InceptionV3_notop.safetensors", "inceptionv3",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite,
    )


def inceptionresnetv2(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Extract InceptionResNetV2 features from an optional weights_path folder.

    Omit weights_path to download MedMLX/radimagenet-mlx. Explicit folders
    must contain RadImageNet-IRV2_notop.safetensors. Input/output behavior matches resnet50.
    """
    return _run_backbone(
        "radimagenet-inceptionresnetv2", "RadImageNet-IRV2_notop.safetensors", "inceptionresnetv2",
        image_path=image_path, output_dir=output_dir,
        weights_path=weights_path, overwrite=overwrite,
    )
