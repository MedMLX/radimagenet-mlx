"""Native Metal execution of four RadImageNet Keras graphs and supplied heads."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from medmlx_core.errors import AssetNotReadyError

from radimagenet_mlx._runtime import import_mlx
from radimagenet_mlx._typing import ArrayRuntime, FloatArray, HostArray
from radimagenet_mlx.checkpoint import load_checkpoint
from radimagenet_mlx.features import MeanRuntime, native_average_pool
from radimagenet_mlx.graph import (
    ARCHITECTURES,
    Activation,
    Architecture,
    BatchNorm,
    Conv,
    Dense,
    Dropout,
    GlobalPool,
    Inference,
    Input,
    Merge,
    Node,
    Pad,
    Padding,
    Pool,
    ResidualScale,
)

if TYPE_CHECKING:
    from mlx.core import array

mx = import_mlx()
OUTPUT_LAYER = ARCHITECTURES[Architecture.RESNET50].spatial


class PoolFunction(Protocol):
    """An MLX pooling layer that maps one NHWC tensor to one tensor."""

    def __call__(self, value: array) -> array:
        """Apply a pooling layer to an MLX array."""
        ...


class PoolRuntime(Protocol):
    """Subset of ``mlx.nn`` used to construct average and max pooling layers."""

    def AvgPool2d(self, kernel: tuple[int, int], stride: tuple[int, int]) -> PoolFunction:
        """Construct an MLX average-pooling layer."""
        ...

    def MaxPool2d(self, kernel: tuple[int, int], stride: tuple[int, int]) -> PoolFunction:
        """Construct an MLX max-pooling layer."""
        ...


nn = cast(PoolRuntime, import_module("mlx.nn"))


def _activation(value: array, kind: Activation) -> array:
    """Apply the requested output activation to an MLX tensor."""
    if kind == Activation.RELU:
        return mx.maximum(value, 0)
    if kind == Activation.SOFTMAX:
        return mx.softmax(value, axis=-1)
    if kind == Activation.SIGMOID:
        return 1 / (1 + mx.exp(-value))
    return value


def _same_padding_widths(
    value: array, kernel: tuple[int, int], stride: tuple[int, int]
) -> list[tuple[int, int]]:
    """Return spatial TensorFlow ``SAME`` widths, placing the odd pixel after."""
    widths: list[tuple[int, int]] = []
    for size, window, step in zip(value.shape[1:3], kernel, stride, strict=True):
        total = max(((size + step - 1) // step - 1) * step + window - size, 0)
        widths.append((total // 2, total - total // 2))
    return widths


def _same_padding(
    value: array, kernel: tuple[int, int], stride: tuple[int, int], fill: float = 0
) -> array:
    """Pad an NHWC tensor using TensorFlow SAME spatial widths."""
    widths = _same_padding_widths(value, kernel, stride)
    return mx.pad(value, [(0, 0), *widths, (0, 0)], constant_values=fill)


def _pool(value: array, op: Pool) -> array:
    """Apply VALID or TensorFlow-compatible SAME spatial pooling."""
    pool = nn.AvgPool2d(op.kernel, op.stride) if op.average else nn.MaxPool2d(op.kernel, op.stride)
    if op.padding == Padding.VALID:
        return pool(value)
    # TensorFlow excludes padded positions from AvgPool's divisor. MLX's
    # AvgPool includes zeros, so pool a validity mask with the same geometry.
    if not op.average:
        return pool(_same_padding(value, op.kernel, op.stride, -float("inf")))
    valid = mx.ones((*value.shape[:3], 1), dtype=value.dtype)
    numerator = pool(_same_padding(value, op.kernel, op.stride))
    denominator = pool(_same_padding(valid, op.kernel, op.stride))
    return numerator / denominator


class RadImageNetMlx:
    """Load a safetensors graph without importing TensorFlow.

    Inputs are prepared NHWC float32 arrays ``(B, H, W, 3)``: this class does no
    preprocessing, so pass the output of ``radimagenet_mlx.prepare_image`` (BGR,
    mean-subtracted, scaled by 1/255). ``features`` always extracts the backbone vector;
    ``predict`` requires a supplied Dense classifier. A no-top checkpoint never
    constructs or invents a classifier. Direct array inputs are cast to float32;
    the executor does not check for non-finite values or overflow during that cast.

    ``checkpoint`` is a local safetensors file with graph metadata.
    If ``architecture`` is given, the checkpoint must be that family. Raises
    ``AssetNotReadyError`` for an invalid checkpoint or a different family.
    """

    def __init__(self, checkpoint: Path, *, architecture: Architecture | None = None) -> None:
        """Load a supported graph and prepare its MLX weights and normalization scales."""
        saved = load_checkpoint(checkpoint)
        if architecture is not None and saved.graph.architecture != architecture:
            raise AssetNotReadyError(f"Expected {architecture}, got {saved.graph.architecture}")
        self.graph = saved.graph
        self.weights = {
            f"{weight.node}/{weight.parameter}": mx.array(weight.value) for weight in saved.weights
        }
        # Checkpoint parameters are fixed inference state. Preserve the original
        # subtract/scale/add order, while evaluating constant BN scales only once.
        self._batch_norm_scales: dict[str, array] = {}
        self._consumers = {node.name: 0 for node in self.graph.nodes}
        for node in self.graph.nodes:
            for edge in node.inputs:
                self._consumers[edge.source] += 1
            if isinstance(node.operation, BatchNorm):
                scale = mx.rsqrt(
                    self.weights[f"{node.name}/moving_variance"] + node.operation.epsilon
                )
                if node.operation.scale:
                    scale = scale * self.weights[f"{node.name}/gamma"]
                self._batch_norm_scales[node.name] = scale
        cast(ArrayRuntime, mx).eval(*self.weights.values(), *self._batch_norm_scales.values())

    @property
    def has_classifier(self) -> bool:
        """Report whether the saved graph contains a Dense classifier."""
        return any(isinstance(node.operation, Dense) for node in self.graph.nodes)

    def _batch_norm(self, value: array, name: str, op: BatchNorm) -> array:
        """Apply the saved batch-normalization statistics and optional offset."""
        weights = self.weights
        result = (value - weights[f"{name}/moving_mean"]) * self._batch_norm_scales[name]
        return result + weights[f"{name}/beta"] if op.center else result

    def _apply(self, node: Node, inputs: list[array]) -> array:
        """Evaluate one supported graph node from its incoming MLX tensors."""
        op, value, name = node.operation, inputs[0], node.name
        if isinstance(op, Conv):
            padding = (0, 0)
            if op.padding == Padding.SAME:
                widths = _same_padding_widths(value, op.kernel, op.stride)
                if all(before == after for before, after in widths):
                    # Convolution can read symmetric padding virtually instead
                    # of materializing an additional activation tensor.
                    padding = (widths[0][0], widths[1][0])
                else:
                    value = mx.pad(value, [(0, 0), *widths, (0, 0)])
            value = mx.conv2d(
                value, self.weights[f"{name}/kernel"], stride=op.stride, padding=padding
            )
            if op.bias:
                value = value + self.weights[f"{name}/bias"]
            return _activation(value, op.activation)
        if isinstance(op, BatchNorm):
            return self._batch_norm(value, name, op)
        if isinstance(op, Activation):
            return _activation(value, op)
        if isinstance(op, Pool):
            return _pool(value, op)
        if isinstance(op, Pad):
            return mx.pad(value, [(0, 0), *op.widths, (0, 0)])
        if isinstance(op, Merge):
            return mx.concatenate(inputs, axis=-1) if op.concatenate else sum(inputs[1:], value)
        if isinstance(op, ResidualScale):
            return value + inputs[1] * op.scale
        if isinstance(op, GlobalPool):
            reduction = mx.mean if op.average else mx.max
            return reduction(value, axis=(1, 2), keepdims=op.keepdims)
        if isinstance(op, Dense):
            value = value @ self.weights[f"{name}/kernel"]
            if op.bias:
                value = value + self.weights[f"{name}/bias"]
            return _activation(value, op.activation)
        if isinstance(op, Dropout):
            return value
        raise ValueError(f"Unexpected graph operation: {node}")

    def _forward(self, images: HostArray, target: str) -> tuple[array, array]:
        """Build the requested graph prefix, retaining only live branch references."""
        if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape) <= 0:
            raise ValueError(f"RadImageNet MLX expects BxHxWx3, got {images.shape}")
        spec = self.graph.nodes[0].operation
        if not isinstance(spec, Input):
            raise ValueError("Graph must begin with its input node")
        if spec.batch is not None and spec.batch != images.shape[0]:
            raise ValueError(f"Checkpoint batch dimension is fixed: {spec.batch}")
        minimum = ARCHITECTURES[self.graph.architecture].minimum_size
        if min(images.shape[1:3]) < minimum:
            raise ValueError(f"{self.graph.architecture} requires spatial dimensions >= {minimum}")
        if any(
            size is not None and size != actual
            for size, actual in zip(spec.size, images.shape[1:3], strict=True)
        ):
            raise ValueError(f"Checkpoint input spatial shape is fixed: {spec.size}")
        values = {self.graph.input: mx.array(np.asarray(images, dtype=np.float32))}
        consumers = self._consumers.copy()
        spatial: array | None = None
        for node in self.graph.nodes[1:]:
            value = self._apply(node, [values[edge.source] for edge in node.inputs])
            values[node.name] = value
            if node.name == self.graph.spatial:
                spatial = value
            for edge in node.inputs:
                consumers[edge.source] -= 1
                if consumers[edge.source] == 0:
                    del values[edge.source]
            # MLX owns execution liveness after these Python references are
            # dropped. Evaluate at the public output boundary, without forcing
            # unfinished residual/concatenation branches at fixed node intervals.
            if node.name == target:
                if spatial is None:
                    raise ValueError("Saved output precedes its backbone features")
                return value, spatial
        raise ValueError(f"Missing output tensor: {target}")

    def spatial_features(self, images: HostArray) -> FloatArray:
        """Return the last spatial backbone tensor, float32 ``(B, H, W, C)``.

        ``C`` is 1024, 1536 or 2048 depending on the architecture.
        """
        output = self._forward(images, self.graph.spatial)[0]
        cast(ArrayRuntime, mx).eval(output)
        return np.asarray(output, dtype=np.float32).copy()

    def extract(self, images: HostArray) -> Inference:
        """Evaluate backbone features and a supplied classifier in one graph pass.

        Returns spatial ``(B, H, W, C)`` and pooled ``(B, C)`` float32 arrays, plus the
        classifier output ``(B, classes)`` when the checkpoint has one (else ``None``).
        """
        target = self.graph.output if self.has_classifier else self.graph.spatial
        output, spatial = self._forward(images, target)
        pooled = native_average_pool(spatial, cast(MeanRuntime, mx))
        cast(ArrayRuntime, mx).eval(spatial, pooled, output)
        predictions = np.asarray(output, dtype=np.float32).copy() if self.has_classifier else None
        return Inference(
            np.asarray(spatial, dtype=np.float32).copy(),
            np.asarray(pooled, dtype=np.float32).copy(),
            predictions,
        )

    def features(self, images: HostArray) -> FloatArray:
        """Return global-average-pooled backbone features, float32 ``(B, C)``."""
        value = native_average_pool(
            self._forward(images, self.graph.spatial)[0], cast(MeanRuntime, mx)
        )
        cast(ArrayRuntime, mx).eval(value)
        return np.asarray(value, dtype=np.float32).copy()

    def predict(self, images: HostArray) -> FloatArray:
        """Return the supplied classifier's output, float32 ``(B, classes)``.

        Raises:
            AssetNotReadyError: If the checkpoint has no classifier (no-top).
        """
        if not self.has_classifier:
            raise AssetNotReadyError("This no-top checkpoint has no supplied trained classifier")
        output = self._forward(images, self.graph.output)[0]
        cast(ArrayRuntime, mx).eval(output)
        return np.asarray(output, dtype=np.float32).copy()

    def __call__(self, images: HostArray) -> FloatArray:
        """Return classifier output if the checkpoint has one, else pooled features."""
        return self.predict(images) if self.has_classifier else self.features(images)


class RadImageNetResNet50Mlx(RadImageNetMlx):
    """``RadImageNetMlx`` that rejects any checkpoint other than ResNet50."""

    def __init__(self, checkpoint: Path) -> None:
        """Load a checkpoint and require the ResNet50 architecture."""
        super().__init__(checkpoint, architecture=Architecture.RESNET50)
