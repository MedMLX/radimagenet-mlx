"""Pure NumPy/Pillow parity against recorded TensorFlow/Keras outputs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from _reference_checkpoint import load_reference

from radimagenet_mlx import global_average_pool, prepare_image
from radimagenet_mlx._typing import FloatArray

FIXTURES = Path(__file__).parent / "fixtures"
INPUTS = load_reference(FIXTURES / "keras_preprocessing.npz")


@pytest.mark.parametrize("case", [str(case) for case in INPUTS["cases"]])
def test_prepare_image_matches_keras_directory_iterator(tmp_path: Path, case: str) -> None:
    """Compare image conversion and resizing with recorded Keras batches."""
    path = tmp_path / f"{case}.png"
    path.write_bytes(INPUTS[f"png::{case}"].tobytes())
    actual = prepare_image(path)
    expected = INPUTS[f"prepared::{case}"]
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape == (1, 224, 224, 3)
    assert np.array_equal(actual, expected), case


@pytest.mark.metal
@pytest.mark.parametrize("case", ["square", "rectangular", "singleton"])
def test_global_average_pool_matches_keras(case: str) -> None:
    """Compare spatial global-average pooling with recorded Keras outputs."""
    saved = load_reference(FIXTURES / "keras_pooling.npz")
    shape = tuple(saved[f"shape::{case}"])
    spatial = np.random.default_rng(saved[f"seed::{case}"].item()).uniform(0, 1, shape)
    actual = global_average_pool(spatial.astype(np.float32))
    expected = saved[f"pooled::{case}"]
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    # Different FP32 summation orders: observed max 6 ULP / 2.981e-7 absolute.
    # Bound: 5e-7 relative + 1e-7 absolute.
    np.testing.assert_allclose(actual, cast(FloatArray, expected), rtol=5e-7, atol=1e-7)

