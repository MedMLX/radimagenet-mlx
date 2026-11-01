"""Strict, non-executable checkpoint loading for RadImageNet graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from medmlx_core.errors import AssetNotReadyError

from radimagenet_mlx._runtime import import_mlx
from radimagenet_mlx._typing import FloatArray, HostArray, JsonValue
from radimagenet_mlx.graph import (
    BatchNorm,
    Conv,
    Dense,
    GlobalPool,
    Graph,
    Input,
    Merge,
    Node,
    parse_graph,
)

if TYPE_CHECKING:
    from mlx.core import array


class SafetensorsRuntime(Protocol):
    """Subset of the MLX module used to read safetensors weights and metadata."""

    def load(
        self, file: str, /, format: Literal["safetensors"], return_metadata: Literal[True]
    ) -> tuple[dict[str, array], dict[str, object]]:
        """Load named tensors and metadata from a safetensors file."""
        ...


@dataclass(frozen=True)
class Weight:
    """One parameter as float32 in MLX layout (Conv2D kernels OHWI, Dense kernels unchanged)."""

    node: str
    parameter: str
    value: FloatArray


@dataclass(frozen=True)
class Checkpoint:
    """A parsed graph with its validated float32 parameters in MLX layout."""

    graph: Graph
    weights: tuple[Weight, ...]


def _weight_shapes(node: Node, channels: int) -> dict[str, tuple[int, ...]]:
    """Expected parameter shapes of ``node`` given its ``channels`` input width.

    Conv kernels are in Keras HWIO order here; ``_load_safetensors`` maps them to the
    stored OHWI layout.
    """
    op = node.operation
    if isinstance(op, Conv):
        return {
            "kernel": (*op.kernel, channels, op.channels),
            **({"bias": (op.channels,)} if op.bias else {}),
        }
    if isinstance(op, Dense):
        return {"kernel": (channels, op.units), **({"bias": (op.units,)} if op.bias else {})}
    if isinstance(op, BatchNorm):
        names = ["moving_mean", "moving_variance"]
        if op.scale:
            names.append("gamma")
        if op.center:
            names.append("beta")
        return {name: (channels,) for name in names}
    return {}


def _shapes(graph: Graph) -> dict[str, dict[str, tuple[int, ...]]]:
    """Propagate channel widths and tensor ranks through the graph.

    Returns each node's expected parameter shapes. Raises ``ValueError`` for a
    Dense layer that does not consume pooled features, an incompatible merge, or
    a spatial tensor whose width differs from the architecture's.
    """
    channels: dict[str, int] = {}
    ranks: dict[str, int] = {}
    expected: dict[str, dict[str, tuple[int, ...]]] = {}
    for node in graph.nodes:
        incoming = [channels[edge.source] for edge in node.inputs]
        ranks_in = [ranks[edge.source] for edge in node.inputs]
        count = incoming[0] if incoming else 3
        op = node.operation
        if isinstance(op, Input):
            rank = 4
        elif isinstance(op, GlobalPool):
            rank = 4 if op.keepdims else 2
        else:
            rank = ranks_in[0]
        if isinstance(op, Dense) and rank != 2:
            raise ValueError(f"Classifier Dense must consume pooled features: {node.name}")
        if isinstance(op, Merge) and (
            len(set(ranks_in)) != 1 or (not op.concatenate and len(set(incoming)) != 1)
        ):
            raise ValueError(f"Incompatible tensor merge: {node.name}")
        expected[node.name] = _weight_shapes(node, count)
        channels[node.name] = (
            op.channels
            if isinstance(op, Conv)
            else op.units
            if isinstance(op, Dense)
            else sum(incoming)
            if isinstance(op, Merge) and op.concatenate
            else count
        )
        ranks[node.name] = rank
    if channels[graph.spatial] != graph.channels or ranks[graph.spatial] != 4:
        raise ValueError("Checkpoint spatial feature width disagrees with its architecture")
    return expected


def _load_safetensors(path: Path) -> Checkpoint:
    """Load declarative graph metadata and finite FP32 tensors in MLX layout."""
    mx = cast(SafetensorsRuntime, import_mlx())
    try:
        tensors, metadata = mx.load(str(path), format="safetensors", return_metadata=True)
        raw = metadata.get("model_config")
        if not isinstance(raw, str):
            raise ValueError("missing model_config metadata")
        graph = parse_graph(cast(JsonValue, json.loads(raw)))
        expected = _shapes(graph)
        names = {f"{node}/{parameter}" for node, params in expected.items() for parameter in params}
        if tensors.keys() != names:
            raise ValueError("safetensors weight names differ from the graph")
        operations = {node.name: node.operation for node in graph.nodes}
        weights: list[Weight] = []
        for name, value in tensors.items():
            node, parameter = name.rsplit("/", 1)
            shape = expected[node][parameter]
            # Conv2D kernels are stored as OHWI; the graph records Keras HWIO shapes.
            if isinstance(operations[node], Conv) and parameter == "kernel":
                shape = (shape[3], shape[0], shape[1], shape[2])
            host = cast(HostArray, np.asarray(value))
            if host.shape != shape or host.dtype != np.float32:
                raise ValueError(f"Invalid safetensors parameter: {name}")
            values = cast(FloatArray, host)
            if not np.isfinite(values).all() or (
                parameter == "moving_variance" and np.any(values < 0)
            ):
                raise ValueError(f"Invalid safetensors parameter: {name}")
            weights.append(Weight(node, parameter, values))
        return Checkpoint(graph, tuple(weights))
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
        raise AssetNotReadyError(f"Invalid RadImageNet safetensors checkpoint: {error}") from error


def load_checkpoint(path: Path) -> Checkpoint:
    """Load a safetensors graph and its exact, finite float32 parameters.

    Graph metadata describes a supported Keras architecture; tensors use MLX
    layout. The graph and parameters are validated, but source identity and
    provenance are not authenticated.

    Raises:
        AssetNotReadyError: If the file is not a valid supported checkpoint.
    """
    if path.suffix != ".safetensors":
        raise AssetNotReadyError(f"RadImageNet requires a safetensors checkpoint: {path}")
    return _load_safetensors(path)
