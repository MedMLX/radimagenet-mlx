"""Architecture-specific admission through the canonical native graph executor."""

from pathlib import Path

from radimagenet_mlx.graph import Architecture
from radimagenet_mlx.mlx_model import RadImageNetMlx


class RadImageNetInceptionV3Mlx(RadImageNetMlx):
    """``RadImageNetMlx`` that rejects any checkpoint other than InceptionV3."""

    def __init__(self, checkpoint: Path) -> None:
        """Load a checkpoint and require the InceptionV3 architecture."""
        super().__init__(checkpoint, architecture=Architecture.INCEPTIONV3)
