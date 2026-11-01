"""Architecture-specific admission through the canonical native graph executor."""

from pathlib import Path

from radimagenet_mlx.graph import Architecture
from radimagenet_mlx.mlx_model import RadImageNetMlx


class RadImageNetInceptionResNetV2Mlx(RadImageNetMlx):
    """``RadImageNetMlx`` that rejects any checkpoint other than InceptionResNetV2."""

    def __init__(self, checkpoint: Path) -> None:
        """Load a checkpoint and require the InceptionResNetV2 architecture."""
        super().__init__(checkpoint, architecture=Architecture.INCEPTIONRESNETV2)
