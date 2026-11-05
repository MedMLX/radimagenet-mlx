"""End-to-end ResNet50 runner against upstream Keras references, and argument rejection."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from _reference_checkpoint import load_reference
from medmlx_core.errors import InvalidInputError

from radimagenet_mlx import medmlx
from radimagenet_mlx._typing import FloatArray

FIXTURES = Path(__file__).parent / "fixtures"
INPUTS = load_reference(FIXTURES / "keras_preprocessing.npz")


def write_png(directory: Path, case: str) -> Path:
    """Write one recorded image fixture and return its path."""
    path = directory / f"{case}.png"
    path.write_bytes(INPUTS[f"png::{case}"].tobytes())
    return path


@pytest.mark.metal
@pytest.mark.parametrize(
    ("case", "fixture"),
    [("rgb", "keras_resnet50_runner"), ("slice_gray", "keras_resnet50_runner_gray")],
)
def test_resnet50_runner_matches_keras(
    tmp_path: Path, checkpoint: Path, case: str, fixture: str
) -> None:
    """Pooled and spatial features written by the runner match seeded Keras recordings."""
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "RadImageNet-ResNet50_notop.safetensors").symlink_to(checkpoint)
    output = tmp_path / "out"
    output.mkdir()
    result = medmlx.resnet50(
        image_path=str(write_png(tmp_path, case)), output_dir=str(output),
        weights_path=str(folder),
    )
    expected = load_reference(FIXTURES / f"{fixture}.npz")
    outputs = result["outputs"]
    for path, key in ((outputs["features_npy"], "pooled"),
                      (outputs["spatial_features_npy"], "spatial")):
        actual = np.load(path, allow_pickle=False)
        assert actual.dtype == np.float32
        # Independent FP32 Conv/BN kernels accumulate rounding through 50 layers.
        # Bound: 2e-5 absolute + 2e-4 relative.
        np.testing.assert_allclose(actual, cast(FloatArray, expected[key]), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "problem",
    ["missing_image", "unknown_backbone", "overwrite_not_bool", "existing_output",
     "output_links_image"],
)
def test_runner_rejects_invalid_arguments_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
) -> None:
    """Reject each invalid argument category before any weights are downloaded."""

    def no_download(model_id: str) -> Path:
        """Fail the test if the runner reaches the weights download."""
        raise AssertionError(f"downloaded {model_id}")

    monkeypatch.setattr(medmlx, "weights_dir", no_download)
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
    with pytest.raises(InvalidInputError):
        if problem == "unknown_backbone":
            medmlx.run(
                image_path=image_path, weights_path=image_path, output_dir=str(output),
                backbone="resnet18",
            )
        else:
            medmlx.resnet50(image_path=image_path, output_dir=str(output), overwrite=overwrite)
