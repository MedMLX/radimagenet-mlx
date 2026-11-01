"""Apple Silicon admission before the shared Metal runtime is loaded."""

from __future__ import annotations

import platform

from medmlx_core.errors import MissingDependencyError
from medmlx_core.runtime import import_mlx as import_core_mlx

_RUNTIME_HINT = "Install medmlx-core and use macOS arm64 with an available Metal GPU."


def import_mlx():
    """Require macOS arm64, then let medmlx-core select and verify Metal."""

    system, machine = platform.system(), platform.machine()
    if (system, machine) != ("Darwin", "arm64"):
        raise MissingDependencyError(
            f"RadImageNet requires macOS arm64 with MLX on Metal; got {system} {machine}",
            hint=_RUNTIME_HINT,
        )
    try:
        import_core_mlx()
        import mlx.core as mx

        return mx
    except MissingDependencyError as error:
        raise MissingDependencyError(
            f"RadImageNet requires MLX on Metal: {error}", hint=_RUNTIME_HINT,
        ) from error
