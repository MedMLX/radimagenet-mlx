"""Architecture-specific admission through the canonical native graph executor."""

from pathlib import Path

from radimagenet_mlx.graph import Architecture
from radimagenet_mlx.mlx_model import RadImageNetMlx


class RadImageNetDenseNet121Mlx(RadImageNetMlx):
    """``RadImageNetMlx`` that rejects any checkpoint other than DenseNet121."""

    def __init__(self, checkpoint: Path) -> None:
        """Load a checkpoint and require the DenseNet121 architecture."""
        super().__init__(checkpoint, architecture=Architecture.DENSENET121)
