from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _reference_checkpoint import load_reference, write_native_checkpoint

if TYPE_CHECKING:
    from radimagenet_mlx import RadImageNetResNet50Mlx

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The seeded full ResNet50 native safetensors checkpoint recorded with the encoder fixtures."""

    fixture = load_reference(FIXTURES / "keras_checkpoint.npz")
    config = json.loads(str(fixture["model_config"]))
    specs = json.loads(str(fixture["weight_specs"]))
    seed = int(fixture["weight_seed"])
    path = tmp_path_factory.mktemp("radimagenet") / "resnet50.safetensors"
    write_native_checkpoint(path, config, specs, seed)
    return path


@pytest.fixture(scope="session")
def encoder(checkpoint: Path) -> RadImageNetResNet50Mlx:
    """Load the seeded ResNet50 checkpoint for Metal tests."""
    from radimagenet_mlx import RadImageNetResNet50Mlx

    return RadImageNetResNet50Mlx(checkpoint)
