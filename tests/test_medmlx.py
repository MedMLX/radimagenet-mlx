"""End-to-end ResNet50 extraction against upstream Keras references, and argument rejection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn, cast

import medmlx_core.weights as core_weights
import numpy as np
import pytest
from _reference_checkpoint import load_reference, write_native_checkpoint
from medmlx_core.errors import InvalidInputError
from medmlx_core.outputs import validate_outputs

from radimagenet_mlx import medmlx
from radimagenet_mlx._typing import FloatArray, ModelConfig, WeightSpec
from radimagenet_mlx.graph import Architecture

FIXTURES = Path(__file__).parent / "fixtures"
INPUTS = load_reference(FIXTURES / "keras_preprocessing.npz")
MODEL_IDS: dict[Architecture, medmlx.ModelId] = {
    architecture: cast(medmlx.ModelId, model)
    for model, architecture in medmlx.ARCHITECTURES.items()
}


def write_png(directory: Path, case: str) -> Path:
    """Write one recorded image fixture and return its path."""
    path = directory / f"{case}.png"
    path.write_bytes(INPUTS[f"png::{case}"].tobytes())
    return path


@pytest.mark.metal
@pytest.mark.parametrize("output_selection", ["all", "pooled"])
@pytest.mark.parametrize(
    ("case", "fixture"),
    [("rgb", "keras_resnet50_runner"), ("slice_gray", "keras_resnet50_runner_gray")],
)
def test_resnet50_runner_matches_keras(
    tmp_path: Path, checkpoint: Path, case: str, fixture: str, output_selection: str,
) -> None:
    """Pooled and spatial features written by the runner match seeded Keras recordings."""
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "RadImageNet-ResNet50_notop.safetensors").symlink_to(checkpoint)
    output = tmp_path / "out"
    output.mkdir()
    result = medmlx.extract(
        model="radimagenet-resnet50", image_path=str(write_png(tmp_path, case)),
        output_dir=str(output), weights_path=str(folder),
        output_selection=cast(medmlx.OutputSelection, output_selection),
    )
    expected = load_reference(FIXTURES / f"{fixture}.npz")
    outputs = result["outputs"]
    paths, artifacts = validate_outputs(result)
    assert set(artifacts) == set(paths) == set(outputs)
    assert artifacts["metadata"]["kind"] == "report"
    assert {artifacts[role]["kind"] for role in outputs if role != "metadata"} == {
        "feature_tensor"
    }
    assert {entry["coordinate_space"] for entry in artifacts.values()} == {"not_applicable"}
    comparisons = [(outputs["features"], "pooled")]
    if output_selection == "all":
        assert "spatial_features" in outputs
        comparisons.append((outputs["spatial_features"], "spatial"))
    else:
        assert set(outputs) == {"features", "metadata"}
    for path, key in comparisons:
        actual = np.load(path, allow_pickle=False)
        assert actual.dtype == np.float32
        # Independent FP32 Conv/BN kernels accumulate rounding through 50 layers.
        # Bound: 2e-5 absolute + 2e-4 relative.
        np.testing.assert_allclose(actual, cast(FloatArray, expected[key]), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "problem",
    ["missing_image", "unknown_model", "overwrite_not_bool", "existing_output",
     "output_links_image", "unreadable_image", "truncated_image", "output_selection"],
)
def test_runner_rejects_invalid_arguments_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
) -> None:
    """Reject each invalid argument category before any weights are downloaded."""

    def no_download(model_id: str, **_options: object) -> Path:
        """Fail the test if the runner reaches the weights download."""
        raise AssertionError(f"downloaded {model_id}")

    monkeypatch.setattr(core_weights, "weights_dir", no_download)
    image, output = write_png(tmp_path, "gray"), tmp_path / "out"
    output.mkdir()
    image_path, overwrite = str(image), False
    if problem == "missing_image":
        image_path = str(tmp_path / "absent.png")
    elif problem == "overwrite_not_bool":
        overwrite = cast(bool, "false")
    elif problem == "existing_output":
        (output / "features.json").write_bytes(b"stale")
    elif problem == "output_links_image":
        (output / "features.json").symlink_to(image)
        overwrite = True
    elif problem == "unreadable_image":
        image.write_bytes(b"not an image")
    elif problem == "truncated_image":
        encoded = INPUTS["png::rgb"].tobytes()
        image.write_bytes(encoded[:len(encoded) // 2])
    model = cast(medmlx.ModelId, "radimagenet-resnet18" if problem == "unknown_model" else
                 "radimagenet-resnet50")
    selection = cast(
        medmlx.OutputSelection, "unknown" if problem == "output_selection" else "all"
    )
    with pytest.raises(InvalidInputError):
        medmlx.extract(
            model=model, image_path=image_path, output_dir=str(output), overwrite=overwrite,
            output_selection=selection,
        )


def _application_checkpoint(
    path: Path, family: Architecture, *, fixed_shape: bool, batch: int | None = None,
) -> None:
    """Reconstruct a classifier fixture with the declared fixed or dynamic input axes."""
    reference = (
        FIXTURES.parent.parent / "work/real-reference" / f"application_{family}_classifier.npz"
    )
    with np.load(reference, allow_pickle=False) as saved:
        config = cast(ModelConfig, json.loads(saved["model_config"].item()))
        settings = cast(dict[str, object], config["config"]["layers"][0]["config"])
        size = list(saved["images"].shape[1:3]) if fixed_shape else [None, None]
        settings["batch_shape"] = [batch, *size, 3]
        write_native_checkpoint(
            path, config, cast(list[WeightSpec], json.loads(saved["weight_specs"].item())),
            int(saved["seed"].item()),
        )


@pytest.mark.metal
@pytest.mark.parametrize("family", list(Architecture))
@pytest.mark.parametrize("fixed_shape", [False, True], ids=["dynamic-square", "fixed-rectangle"])
def test_pooled_batch_session_matches_full_features(
    family: Architecture, fixed_shape: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All backbones preserve pooled vectors, classifier defaults, batch tails and model reuse."""
    from radimagenet_mlx import mlx_model

    folder = tmp_path / "weights"
    folder.mkdir()
    checkpoint = folder / medmlx.CHECKPOINT_FILES[family]
    _application_checkpoint(checkpoint, family, fixed_shape=fixed_shape)
    images = [str(write_png(tmp_path, case)) for case in ("rgb", "slice_gray", "gray")]
    folders = [str(tmp_path / f"image-{i}") for i in range(len(images))]
    encoder = mlx_model.RadImageNetMlx(checkpoint)
    prepared = np.concatenate([
        medmlx.prepare_image(Path(path), encoder.graph.input_size) for path in images
    ])
    expected = encoder.extract(prepared)
    np.testing.assert_array_equal(encoder.features(prepared), expected.features)
    with medmlx.load_model(model=MODEL_IDS[family], weights_path=folder) as session:
        full = list(session.extract_batch(image_paths=images, output_dirs=folders, batch_size=2))
        for index, result in enumerate(full):
            assert result["details"]["schema_version"] == 3
            assert set(validate_outputs(result).artifacts) == set(result["outputs"])
            assert set(result["outputs"]) == {
                "features", "spatial_features", "metadata", "predictions",
            }
            np.testing.assert_allclose(
                np.load(result["outputs"]["features"]), expected.features[index:index + 1],
                rtol=2e-4, atol=2e-5,
            )

        def no_reload(*_args: object, **_kwargs: object) -> NoReturn:
            raise AssertionError("a loaded model reloaded its network")

        monkeypatch.setattr(mlx_model, "RadImageNetMlx", no_reload)
        pooled = list(session.extract_batch(
            image_paths=images, output_dirs=folders, batch_size=2,
            output_selection="pooled", overwrite=True,
        ))
        for index, result in enumerate(pooled):
            assert set(result["outputs"]) == {"features", "metadata"}
            assert result["details"].get("output_selection") == "pooled"
            assert "spatial_shape" not in result["details"]
            assert "classifier_output" not in result["details"]
            assert not (Path(folders[index]) / "spatial_features.npy").exists()
            assert not (Path(folders[index]) / "predictions.npy").exists()
            np.testing.assert_allclose(
                np.load(result["outputs"]["features"]), expected.features[index:index + 1],
                rtol=2e-4, atol=2e-5,
            )
    with pytest.raises(RuntimeError, match="closed"):
        session.extract(image_path=images[0], output_dir=str(tmp_path / "closed"))


@pytest.mark.metal
def test_batch_runner_honors_fixed_checkpoint_batch(tmp_path: Path) -> None:
    """A fixed batch is used unchanged, and incompatible cohort tails are rejected."""
    folder = tmp_path / "weights"
    folder.mkdir()
    checkpoint = folder / medmlx.CHECKPOINT_FILES[Architecture.RESNET50]
    _application_checkpoint(checkpoint, Architecture.RESNET50, fixed_shape=True, batch=2)
    images = [str(write_png(tmp_path, case)) for case in ("rgb", "gray")]
    with medmlx.load_model(model="radimagenet-resnet50", weights_path=str(folder)) as loaded:
        results = list(loaded.extract_batch(
            image_paths=images, output_dirs=[str(tmp_path / f"fixed-{i}") for i in range(2)],
            batch_size=3, output_selection="pooled",
        ))
        assert len(results) == 2
        assert np.load(results[0]["outputs"]["features"]).shape == (1, 2048)
        with pytest.raises(InvalidInputError, match="fixed checkpoint batch"):
            list(loaded.extract_batch(
                image_paths=images[:1], output_dirs=[str(tmp_path / "tail")], batch_size=3,
            ))


@pytest.mark.parametrize("problem", ["length", "duplicate", "input_alias", "output_alias", "bound"])
def test_batch_paths_reject_cross_job_collisions_before_weights(
    tmp_path: Path, problem: str,
) -> None:
    """One cohort job cannot overwrite another input or output, including hard links."""
    images = [write_png(tmp_path, case) for case in ("rgb", "gray")]
    folders = [tmp_path / f"output-{i}" for i in range(2)]
    for folder in folders:
        folder.mkdir()
    if problem == "length":
        folders.pop()
    elif problem == "duplicate":
        folders[1] = folders[0]
    elif problem == "input_alias":
        (folders[0] / "features.npy").hardlink_to(images[1])
    elif problem == "output_alias":
        (folders[0] / "features.npy").write_bytes(b"previous features")
        (folders[1] / "spatial_features.npy").hardlink_to(folders[0] / "features.npy")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / medmlx.CHECKPOINT_FILES[Architecture.RESNET50]).write_bytes(b"unread")
    loaded = medmlx.load_model(model="radimagenet-resnet50", weights_path=str(weights))
    with pytest.raises(InvalidInputError):
        list(loaded.extract_batch(
            image_paths=[str(path) for path in images], output_dirs=[str(path) for path in folders],
            batch_size=0 if problem == "bound" else 2, overwrite=True,
        ))
