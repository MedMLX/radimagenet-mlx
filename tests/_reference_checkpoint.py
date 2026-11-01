"""Seeded Keras HDF5 construction shared by fixture generation and tests only."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from radimagenet_mlx._typing import FloatArray, HostArray, ModelConfig, WeightSpec

TENSORFLOW_VERSION = "2.20.0"
KERAS_VERSION = "3.15.0"
KERAS_REVISION = "9d1bbf9aae730708cc6e0d7104860c1605c0e31b"
REFERENCE_PLATFORM = "darwin-arm64"
NORMALIZATION_KEYS = {"gamma:0", "beta:0", "moving_mean:0", "moving_variance:0"}


def load_reference(path: Path) -> dict[str, HostArray]:
    """Load a qualified upstream recording; never silently skip missing goldens."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream fixture {path}; run scripts/make_keras_fixtures.py"
        )
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        assert metadata["schema_version"] == 3
        assert metadata["reference_platform"] == REFERENCE_PLATFORM
        assert metadata["keras_revision"] == KERAS_REVISION
        assert metadata["keras_version"] == KERAS_VERSION
        assert metadata["tensorflow_version"] == TENSORFLOW_VERSION
        return dict(saved)


def seeded_weights(
    specs: list[WeightSpec], seed: int, normalization_seed: int | None = None,
) -> Iterator[tuple[WeightSpec, FloatArray]]:
    """Yield full FP32 random weights in Keras layout from fixture metadata."""
    rng = np.random.Generator(np.random.PCG64(seed))
    normalization_rng = (
        np.random.default_rng(normalization_seed) if normalization_seed is not None else None
    )
    for spec in specs:
        shape = spec["shape"]
        key = spec["key"]
        if normalization_rng is not None and key in NORMALIZATION_KEYS:
            low, high = (-0.2, 0.2) if key in {"beta:0", "moving_mean:0"} else (0.5, 1.5)
            value = normalization_rng.uniform(low, high, shape)
        elif key == "kernel:0":
            value = rng.normal(0.0, 1.0 / np.sqrt(np.prod(shape[:-1])), shape)
        elif key == "gamma:0":
            value = rng.uniform(0.8, 1.2, shape)
        elif key == "moving_variance:0":
            value = rng.uniform(0.7, 1.3, shape)
        else:
            value = rng.normal(0.0, 0.02, shape)
        yield spec, value.astype(np.float32)


def write_checkpoint(
    path: Path, config: ModelConfig, specs: list[WeightSpec], seed: int,
    *, normalization_seed: int | None = None,
) -> None:
    """Rebuild full FP32 random weights from fixture metadata (no pretrained assets)."""
    with h5py.File(str(path), "w") as handle:
        handle.attrs["model_config"] = json.dumps(config)
        weights = handle.create_group("model_weights")
        for spec, value in seeded_weights(specs, seed, normalization_seed):
            group = weights.require_group(spec["layer"]).require_group(spec["layer"])
            group.create_dataset(spec["key"], data=value)


def write_native_checkpoint(
    path: Path, config: ModelConfig, specs: list[WeightSpec], seed: int,
    *, normalization_seed: int | None = None,
) -> None:
    """Reconstruct the recorded seeded fixture in native tensor layout."""
    from typing import Protocol, cast

    import mlx.core as mx

    class Writer(Protocol):
        def save_safetensors(
            self, path: str, arrays: dict[str, mx.array], metadata: dict[str, str]
        ) -> None:
            """Write named MLX tensors and model metadata as safetensors."""
            ...

    weights: dict[str, mx.array] = {}
    for spec, array in seeded_weights(specs, seed, normalization_seed):
        if array.ndim == 4:
            array = array.transpose(3, 0, 1, 2)
        weights[f"{spec['layer']}/{spec['key'].removesuffix(':0')}"] = mx.array(array)
    cast(Writer, mx).save_safetensors(str(path), weights, {"model_config": json.dumps(config)})
