"""Seeded ResNet50 parity; recorded Keras fixtures need no TensorFlow at test time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from _reference_checkpoint import load_reference, write_native_checkpoint

from radimagenet_mlx._typing import FloatArray

if TYPE_CHECKING:
    from radimagenet_mlx import RadImageNetResNet50Mlx

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.metal


@pytest.mark.parametrize(
    ("case", "rtol", "atol"),
    [
        # Observed max absolute error 3.756e-6 (FP32 Conv/BN rounding); no relative allowance.
        ("square", 0, 4e-6),
        ("small_rectangular", 0, 4e-6),
        # Independent FP32 Conv/BN implementations accumulate rounding through 50 layers.
        ("runner", 2e-4, 2e-5),
        ("rectangular", 2e-4, 2e-5),
    ],
)
def test_resnet50_matches_keras(
    encoder: RadImageNetResNet50Mlx, case: str, rtol: float, atol: float
) -> None:
    """Compare seeded ResNet50 spatial and pooled outputs with Keras recordings."""
    saved = load_reference(FIXTURES / f"keras_resnet50_{case}.npz")
    spatial = encoder.spatial_features(saved["images"])
    pooled = encoder(saved["images"])
    assert spatial.dtype == pooled.dtype == np.float32
    assert spatial.shape == saved["spatial"].shape
    assert pooled.shape == saved["pooled"].shape == (saved["images"].shape[0], 2048)
    np.testing.assert_allclose(spatial, cast(FloatArray, saved["spatial"]), rtol=rtol, atol=atol)
    np.testing.assert_allclose(pooled, cast(FloatArray, saved["pooled"]), rtol=rtol, atol=atol)


def test_resnet50_matches_keras_with_initialized_normalization(tmp_path: Path) -> None:
    """Compare default BN epsilons and seed-0 normalization state with Keras."""
    from radimagenet_mlx import RadImageNetResNet50Mlx

    recipe = load_reference(FIXTURES / "keras_initialized_checkpoint.npz")
    checkpoint = tmp_path / "resnet50.safetensors"
    write_native_checkpoint(
        checkpoint, json.loads(recipe["model_config"].item()),
        json.loads(recipe["weight_specs"].item()), int(recipe["weight_seed"].item()),
        normalization_seed=int(recipe["normalization_seed"].item()),
    )
    fixture = load_reference(FIXTURES / "keras_resnet50_initialized.npz")
    model = RadImageNetResNet50Mlx(checkpoint)
    spatial = model.spatial_features(fixture["images"])
    pooled = model(fixture["images"])
    assert spatial.shape == fixture["spatial"].shape == (2, 2, 2, 2048)
    assert pooled.shape == fixture["pooled"].shape == (2, 2048)
    # Bound for this case: 1e-4 absolute + 1e-4 relative.
    np.testing.assert_allclose(spatial, cast(FloatArray, fixture["spatial"]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pooled, cast(FloatArray, fixture["pooled"]), rtol=1e-4, atol=1e-4)

