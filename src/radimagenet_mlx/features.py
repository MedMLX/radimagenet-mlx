"""One native average-pooling operation, with an explicit NumPy boundary helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from medmlx_core.typing import MlxRuntime

from radimagenet_mlx._runtime import import_mlx
from radimagenet_mlx._typing import FloatArray, RealArray
from radimagenet_mlx.graph import ARCHITECTURES

if TYPE_CHECKING:
    from mlx.core import array

# Name recorded in run metadata for the global-average-pooled feature vector.
POOLED_OUTPUT_LAYER = "avg_pool"


def native_average_pool(spatial: array, runtime: MlxRuntime) -> array:
    """Keras NHWC GlobalAveragePooling2D, evaluated by callers on Metal."""
    return runtime.mean(spatial, axis=(1, 2))


def global_average_pool(spatial_features: RealArray) -> FloatArray:
    """Return float32 ``(B, C)`` native Metal pooling of a ``(B, H, W, C)`` spatial map.

    ``C`` must be one of the architectures' widths (1024, 1536 or 2048) and no axis may
    be empty; otherwise a ``ValueError`` is raised. Values are cast to float32;
    finiteness is not checked before or after that cast.

    MLX loads only when called; importing the helper does not select a backend.
    Unsupported hosts or unavailable Metal fail explicitly.
    """
    if (
        spatial_features.ndim != 4
        or spatial_features.shape[-1] not in {spec.channels for spec in ARCHITECTURES.values()}
        or any(size == 0 for size in spatial_features.shape)
    ):
        raise ValueError(
            f"RadImageNet pooling expects nonempty BxHxWxC, got {spatial_features.shape}"
        )
    mx = import_mlx()
    value = native_average_pool(
        mx.array(np.asarray(spatial_features, dtype=np.float32)), mx
    )
    mx.eval(value)
    return np.asarray(value, dtype=np.float32).copy()
