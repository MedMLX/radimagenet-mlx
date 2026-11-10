"""Independent full-graph goldens, strict loading, legacy Keras 2 graphs and runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest
from _reference_checkpoint import KERAS_VERSION, TENSORFLOW_VERSION, write_native_checkpoint
from medmlx_core.errors import AssetNotReadyError

from radimagenet_mlx._typing import FloatArray, JsonObject, JsonValue, ModelConfig, WeightSpec
from radimagenet_mlx.checkpoint import load_checkpoint
from radimagenet_mlx.graph import Architecture, object_value, tensor_references

if TYPE_CHECKING:
    from mlx.core import array


class SafetensorsWriter(Protocol):
    def save_safetensors(
        self, path: str, tensors: dict[str, array], metadata: dict[str, str]
    ) -> None:
        """Write named tensors and metadata as a safetensors checkpoint."""
        ...


FIXTURES = Path(__file__).parent / "fixtures"
WEIGHTS = {
    "densenet121": "RadImageNet-DenseNet121_notop.safetensors",
    "inceptionv3": "RadImageNet-InceptionV3_notop.safetensors",
    "inceptionresnetv2": "RadImageNet-IRV2_notop.safetensors",
}


def recipe(family: str, suffix: str) -> dict[str, np.ndarray]:
    """Load one recorded Keras application case and verify its reference versions."""
    with np.load(FIXTURES / f"application_{family}_{suffix}.npz", allow_pickle=False) as saved:
        assert saved["schema_version"].item() == 1
        assert saved["tensorflow_version"].item() == TENSORFLOW_VERSION
        assert saved["keras_version"].item() == KERAS_VERSION
        return dict(saved)


def checkpoint(path: Path, saved: dict[str, np.ndarray]) -> None:
    """Reconstruct seeded native weights and graph metadata from a recording."""
    write_native_checkpoint(
        path,
        cast(ModelConfig, json.loads(saved["model_config"].item())),
        cast(list[WeightSpec], json.loads(saved["weight_specs"].item())),
        int(saved["seed"].item()),
    )


def assert_keras_close(actual: FloatArray, expected: np.ndarray) -> None:
    """Apply the FP32 bounds shared with the ResNet50 Keras comparisons."""
    np.testing.assert_allclose(actual, cast(FloatArray, expected), rtol=2e-4, atol=2e-5)


def _replace_names(value: JsonValue, names: dict[str, str]) -> JsonValue:
    """Recursively replace matching strings in a JSON value."""
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, list):
        return [_replace_names(item, names) for item in value]
    if isinstance(value, dict):
        return {key: _replace_names(item, names) for key, item in value.items()}
    return value


def _as_keras2(config: JsonObject) -> JsonObject:
    """Store tensor edges and the nested output as Keras 2 checkpoints do."""
    settings = object_value(config["config"])
    layers = settings["layers"]
    assert isinstance(layers, list)
    for raw in layers:
        layer = object_value(raw)
        edges = tensor_references(layer["inbound_nodes"])
        layer["inbound_nodes"] = [[[edge.source, 0, 0, {}] for edge in edges]] if edges else []
    settings["output_layers"] = [settings["output_layers"]]
    return config


@pytest.mark.metal
@pytest.mark.parametrize("family", list(Architecture))
@pytest.mark.parametrize("suffix", ["spatial", "classifier"])
def test_complete_application_matches_independent_keras(
    tmp_path: Path, family: Architecture, suffix: str,
) -> None:
    """Compare seeded backbones, pooled features and optional classifiers with Keras."""
    from radimagenet_mlx import RadImageNetMlx

    saved = recipe(family, suffix)
    path = tmp_path / "model.safetensors"
    checkpoint(path, saved)
    model = RadImageNetMlx(path, architecture=family)
    images = cast(FloatArray, saved["images"])
    result = model.extract(images)
    backbone = recipe(family, "spatial")["expected"]
    assert result.spatial.dtype == np.float32
    # Appending the head does not reinitialize any loaded backbone parameters.
    assert_keras_close(result.spatial, backbone)
    assert_keras_close(result.features, np.mean(backbone, axis=(1, 2), dtype=np.float32))
    if suffix == "classifier":
        assert result.predictions is not None
        assert_keras_close(result.predictions, saved["expected"])
        assert_keras_close(model.predict(images), saved["expected"])
    else:
        assert result.predictions is None
        with pytest.raises(AssetNotReadyError):
            model.predict(images)


@pytest.mark.metal
@pytest.mark.parametrize(
    "damage",
    ["missing", "nonfinite", "late_inf", "variance", "dtype", "shape", "extra", "metadata"],
)
def test_safetensors_rejects_invalid_checkpoint(tmp_path: Path, damage: str) -> None:
    """Reject malformed graph metadata and missing or invalid tensor parameters."""
    import mlx.core as mx

    path = tmp_path / "model.safetensors"
    checkpoint(path, recipe("resnet50", "spatial"))
    tensors, metadata = cast(
        tuple[dict[str, mx.array], dict[str, str]],
        mx.load(str(path), format="safetensors", return_metadata=True),
    )
    name = "conv1_conv/kernel"
    if damage == "missing":
        del tensors[name]
    elif damage == "nonfinite":
        tensors[name] = mx.full(tensors[name].shape, float("nan"))
    elif damage == "late_inf":
        name = next(reversed(tensors))
        tensors[name] = mx.full(tensors[name].shape, float("inf"))
    elif damage == "variance":
        name = next(key for key in tensors if key.endswith("/moving_variance"))
        tensors[name] = mx.full(tensors[name].shape, -1.0)
    elif damage == "dtype":
        tensors[name] = tensors[name].astype(mx.float16)
    elif damage == "shape":
        tensors[name] = mx.zeros((1,))
    elif damage == "extra":
        tensors["unused/kernel"] = mx.zeros((1,))
    else:
        metadata["model_config"] = "{}"
    cast(SafetensorsWriter, mx).save_safetensors(str(path), tensors, metadata=metadata)
    with pytest.raises(AssetNotReadyError):
        load_checkpoint(path)


@pytest.mark.metal
def test_keras2_residual_lambdas_are_declarative_not_executable(tmp_path: Path) -> None:
    """Load a Keras 2 InceptionResNetV2 graph without executing its Lambda payloads."""
    from radimagenet_mlx import RadImageNetMlx

    saved = recipe("inceptionresnetv2", "spatial")
    config = object_value(cast(JsonValue, json.loads(saved["model_config"].item())))
    layers = object_value(config["config"])["layers"]
    assert isinstance(layers, list)
    renames: dict[str, str] = {}
    for raw in layers:
        layer = object_value(raw)
        if layer["class_name"] != "CustomScaleLayer":
            continue
        inputs = tensor_references(layer["inbound_nodes"])
        name = inputs[1].source.removesuffix("_conv")
        renames[str(layer["name"])] = name
        settings = object_value(layer["config"])
        layer["class_name"] = "Lambda"
        layer["name"] = name
        settings["name"] = name
        settings["arguments"] = {"scale": settings.pop("scale")}
        settings["function"] = "never decode or execute this payload"

    config = _as_keras2(object_value(_replace_names(config, renames)))
    saved["model_config"] = np.asarray(json.dumps(config))
    path = tmp_path / "legacy.safetensors"
    checkpoint(path, saved)
    assert_keras_close(RadImageNetMlx(path).spatial_features(saved["images"]), saved["expected"])


@pytest.mark.metal
def test_keras2_densenet_slash_names_preserve_exact_weights(tmp_path: Path) -> None:
    """Load a Keras 2 DenseNet121 graph with slash-delimited layer names."""
    from radimagenet_mlx import RadImageNetMlx

    saved = recipe("densenet121", "spatial")
    names = {"conv1_conv": "conv1/conv", "conv1_bn": "conv1/bn"}
    config = object_value(cast(JsonValue, json.loads(saved["model_config"].item())))
    specs = cast(JsonValue, json.loads(saved["weight_specs"].item()))
    config = _as_keras2(object_value(_replace_names(config, names)))
    saved["model_config"] = np.asarray(json.dumps(config))
    saved["weight_specs"] = np.asarray(json.dumps(_replace_names(specs, names)))
    path = tmp_path / "legacy-densenet.safetensors"
    checkpoint(path, saved)
    model = RadImageNetMlx(path)
    assert "conv1/conv/kernel" in model.weights
    assert_keras_close(model.spatial_features(saved["images"]), saved["expected"])


@pytest.mark.metal
@pytest.mark.parametrize("runner", list(WEIGHTS))
def test_registered_runner_writes_features(tmp_path: Path, runner: str) -> None:
    """Run each remaining registered backbone end to end on a seeded weights folder."""
    from radimagenet_mlx import medmlx

    image, folder = tmp_path / "input.png", tmp_path / "weights"
    with np.load(FIXTURES / "keras_preprocessing.npz", allow_pickle=False) as payload:
        image.write_bytes(payload["png::rgb"].tobytes())
    folder.mkdir()
    checkpoint(folder / WEIGHTS[runner], recipe(runner, "spatial"))
    result = getattr(medmlx, runner)(
        image_path=str(image), output_dir=str(tmp_path / "out"), weights_path=str(folder),
    )
    assert result["details"]["backbone"] == runner
    spatial = np.load(result["outputs"]["spatial_features_npy"], allow_pickle=False)
    features = np.load(result["outputs"]["features_npy"], allow_pickle=False)
    assert spatial.ndim == 4 and features.shape == (1, spatial.shape[-1])
    assert np.isfinite(features).all()
