"""Native MLX implementation of RadImageNet.

Public names load on first access, so importing the package (or the medmlx
declaration in ``radimagenet_mlx.medmlx``) does not import MLX.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radimagenet_mlx.densenet_model import (
        RadImageNetDenseNet121Mlx as RadImageNetDenseNet121Mlx,
    )
    from radimagenet_mlx.features import POOLED_OUTPUT_LAYER as POOLED_OUTPUT_LAYER
    from radimagenet_mlx.features import global_average_pool as global_average_pool
    from radimagenet_mlx.graph import Architecture as Architecture
    from radimagenet_mlx.inception_resnet_v2_model import (
        RadImageNetInceptionResNetV2Mlx as RadImageNetInceptionResNetV2Mlx,
    )
    from radimagenet_mlx.inception_v3_model import (
        RadImageNetInceptionV3Mlx as RadImageNetInceptionV3Mlx,
    )
    from radimagenet_mlx.medmlx import extract as extract
    from radimagenet_mlx.medmlx import load_model as load_model
    from radimagenet_mlx.medmlx import prepare_image as prepare_image
    from radimagenet_mlx.mlx_model import OUTPUT_LAYER as OUTPUT_LAYER
    from radimagenet_mlx.mlx_model import RadImageNetMlx as RadImageNetMlx
    from radimagenet_mlx.mlx_model import RadImageNetResNet50Mlx as RadImageNetResNet50Mlx

__version__ = "0.3.4"

_EXPORTS = {
    "Architecture": "radimagenet_mlx.graph",
    "RadImageNetMlx": "radimagenet_mlx.mlx_model",
    "OUTPUT_LAYER": "radimagenet_mlx.mlx_model",
    "POOLED_OUTPUT_LAYER": "radimagenet_mlx.features",
    "RadImageNetDenseNet121Mlx": "radimagenet_mlx.densenet_model",
    "RadImageNetInceptionResNetV2Mlx": "radimagenet_mlx.inception_resnet_v2_model",
    "RadImageNetInceptionV3Mlx": "radimagenet_mlx.inception_v3_model",
    "RadImageNetResNet50Mlx": "radimagenet_mlx.mlx_model",
    "global_average_pool": "radimagenet_mlx.features",
    "extract": "radimagenet_mlx.medmlx",
    "load_model": "radimagenet_mlx.medmlx",
    "prepare_image": "radimagenet_mlx.medmlx",
}

__all__ = [
    "OUTPUT_LAYER",
    "POOLED_OUTPUT_LAYER",
    "Architecture",
    "RadImageNetDenseNet121Mlx",
    "RadImageNetInceptionResNetV2Mlx",
    "RadImageNetInceptionV3Mlx",
    "RadImageNetMlx",
    "RadImageNetResNet50Mlx",
    "extract",
    "global_average_pool",
    "load_model",
    "prepare_image",
]


def __getattr__(name: str) -> object:
    """Load and cache a public export when first accessed."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'radimagenet_mlx' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive package discovery."""
    return sorted({*globals(), *_EXPORTS})
