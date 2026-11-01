"""Canonical, safe typed ontology for the four released Keras application graphs.

Only tensor edges and declarative layer configuration are parsed. Executable
Keras Lambda payloads are never decoded, imported, or evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from radimagenet_mlx._typing import FloatArray, JsonObject, JsonValue


class Architecture(StrEnum):
    """Supported RadImageNet backbone families, named as in Keras applications."""

    RESNET50 = "resnet50"
    DENSENET121 = "densenet121"
    INCEPTIONV3 = "inceptionv3"
    INCEPTIONRESNETV2 = "inceptionresnetv2"


class Activation(StrEnum):
    """Supported Keras activation functions; any other name is rejected."""

    LINEAR = "linear"
    RELU = "relu"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


class Padding(StrEnum):
    """TensorFlow 2D padding modes; ``SAME`` pads so every input pixel is covered."""

    VALID = "valid"
    SAME = "same"


@dataclass(frozen=True)
class Input:
    """Input layer.

    ``size`` is ``(height, width)`` and ``batch`` the batch size; ``None`` means dynamic.
    """

    size: tuple[int | None, int | None]
    batch: int | None = None

    def __post_init__(self) -> None:
        """Reject nonpositive fixed input dimensions."""
        if any(value is not None and value <= 0 for value in (*self.size, self.batch)):
            raise ValueError("Input dimensions must be positive or dynamic")


@dataclass(frozen=True)
class Conv:
    """2D convolution with a fused activation; undilated and ungrouped."""

    channels: int
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: Padding
    bias: bool
    activation: Activation

    def __post_init__(self) -> None:
        """Reject nonpositive channel, kernel, or stride dimensions."""
        if min(self.channels, *self.kernel, *self.stride) <= 0:
            raise ValueError("Convolution dimensions must be positive")


@dataclass(frozen=True)
class BatchNorm:
    """Channel-wise batch normalization using saved inference statistics.

    ``epsilon`` stabilizes the variance; ``center`` and ``scale`` toggle the beta
    offset and gamma multiplier.
    """

    epsilon: float
    center: bool
    scale: bool

    def __post_init__(self) -> None:
        """Require a finite, positive normalization epsilon."""
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Batch normalization epsilon must be finite and positive")


@dataclass(frozen=True)
class Pool:
    """Max or average pooling; ``same`` padding follows TensorFlow's placement."""

    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: Padding
    average: bool

    def __post_init__(self) -> None:
        """Reject nonpositive kernel or stride dimensions."""
        if min(*self.kernel, *self.stride) <= 0:
            raise ValueError("Pooling dimensions must be positive")


@dataclass(frozen=True)
class Pad:
    """Zero padding of ``((top, bottom), (left, right))`` pixels on an NHWC tensor."""

    widths: tuple[tuple[int, int], tuple[int, int]]

    def __post_init__(self) -> None:
        """Reject negative padding widths."""
        if min(*self.widths[0], *self.widths[1]) < 0:
            raise ValueError("Padding widths must be nonnegative")


@dataclass(frozen=True)
class Merge:
    """Add (elementwise sum) or, if ``concatenate``, join along the channel axis."""

    concatenate: bool


@dataclass(frozen=True)
class ResidualScale:
    """Inception-ResNet residual merge ``x + scale * branch`` (two inputs: x, branch)."""

    scale: float

    def __post_init__(self) -> None:
        """Require a finite residual scale."""
        if not isfinite(self.scale):
            raise ValueError("Residual scale must be finite")


@dataclass(frozen=True)
class GlobalPool:
    """Pool an NHWC tensor over its height and width axes.

    ``average`` selects the mean over the maximum; ``keepdims`` retains the
    ``(B, 1, 1, C)`` layout instead of producing ``(B, C)``.
    """

    average: bool
    keepdims: bool


@dataclass(frozen=True)
class Dense:
    """Fully connected classifier over pooled features.

    ``units`` is the output width; ``bias`` adds a per-unit offset and
    ``activation`` is fused into the layer.
    """

    units: int
    bias: bool
    activation: Activation

    def __post_init__(self) -> None:
        """Require a positive classifier width."""
        if self.units <= 0:
            raise ValueError("Classifier width must be positive")


@dataclass(frozen=True)
class Dropout:
    """Inference identity; no random operation or training mode."""


type Operation = (
    Input
    | Conv
    | BatchNorm
    | Pool
    | Pad
    | Merge
    | ResidualScale
    | GlobalPool
    | Dense
    | Dropout
    | Activation
)


@dataclass(frozen=True)
class TensorEdge:
    """Reference to the output tensor of the named source layer."""

    source: str


@dataclass(frozen=True)
class Node:
    """One graph layer: a unique name, its declarative operation, and incoming edges.

    ``inputs`` preserves the layer's argument order, which fixes the operand order of
    residual sums and the channel order of concatenations.
    """

    name: str
    operation: Operation
    inputs: tuple[TensorEdge, ...]


@dataclass(frozen=True)
class Graph:
    """A parsed saved model in topological order.

    ``spatial`` names the last spatial (NHWC) tensor and ``channels`` is its width;
    ``output`` is the saved model's output, which may be spatial, pooled or a classifier.
    """

    architecture: Architecture
    nodes: tuple[Node, ...]
    input: str
    output: str
    spatial: str
    channels: int

    @property
    def input_size(self) -> tuple[int, int]:
        """``(height, width)`` from the saved input layer, else the architecture default."""
        spec = self.nodes[0].operation
        if not isinstance(spec, Input):
            raise ValueError("Graph must begin with its input")
        default = ARCHITECTURES[self.architecture].input_size
        return spec.size[0] or default[0], spec.size[1] or default[1]


@dataclass(frozen=True)
class Inference:
    """Float32 outputs of one graph pass.

    ``spatial`` is ``(B, H, W, C)``, ``features`` the global-average-pooled ``(B, C)``
    vector, and ``predictions`` the classifier output ``(B, classes)`` or ``None`` for a
    no-top checkpoint.
    """

    spatial: FloatArray
    features: FloatArray
    predictions: FloatArray | None


def object_value(value: JsonValue) -> JsonObject:
    """Return a JSON object or reject another value type."""
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object in Keras configuration")
    return value


def list_value(value: JsonValue) -> list[JsonValue]:
    """Return a JSON list or reject another value type."""
    if not isinstance(value, list):
        raise ValueError("Expected JSON list in Keras configuration")
    return value


def string(value: JsonValue) -> str:
    """Return a JSON string or reject another value type."""
    if not isinstance(value, str):
        raise ValueError("Expected string in Keras configuration")
    return value


def integer(value: JsonValue) -> int:
    """Return a JSON integer, rejecting booleans and other value types."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Expected integer in Keras configuration")
    return value


def boolean(value: JsonValue) -> bool:
    """Return a JSON boolean or reject another value type."""
    if not isinstance(value, bool):
        raise ValueError("Expected boolean in Keras configuration")
    return value


def number(value: JsonValue) -> float:
    """Convert a JSON integer or float to float, rejecting booleans."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Expected number in Keras configuration")
    return float(value)


def pair(value: JsonValue) -> tuple[int, int]:
    """Expand an integer or parse a two-element spatial pair."""
    if isinstance(value, int):
        return value, value
    values = list_value(value)
    if len(values) != 2:
        raise ValueError("Expected two spatial dimensions")
    return integer(values[0]), integer(values[1])


def tensor_references(value: JsonValue) -> tuple[TensorEdge, ...]:
    """Read Keras 2 node triples or Keras 3 symbolic tensor arguments."""
    if isinstance(value, dict):
        if value.get("class_name") == "__keras_tensor__":
            history = list_value(object_value(value["config"])["keras_history"])
            return _edge(history)
        return tuple(edge for child in value.values() for edge in tensor_references(child))
    if isinstance(value, list):
        if len(value) >= 3 and isinstance(value[0], str) and isinstance(value[1], int):
            return _edge(value)
        return tuple(edge for child in value for edge in tensor_references(child))
    return ()


def _edge(history: list[JsonValue]) -> tuple[TensorEdge, ...]:
    """Parse a Keras tensor history, rejecting shared or multi-output layers."""
    if len(history) < 3:
        raise ValueError("Incomplete Keras tensor reference")
    if integer(history[1]) != 0 or integer(history[2]) != 0:
        raise ValueError("Shared layers and multiple-output tensors are not supported")
    return (TensorEdge(string(history[0])),)


def _input(config: JsonObject) -> Input:
    """Parse a float32 channels-last RGB input definition."""
    if config.get("dtype", "float32") != "float32":
        raise ValueError("Input must use float32")
    shape = list_value(config.get("batch_shape", config.get("batch_input_shape")))
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError("Input must be a float32 NHWC RGB tensor")
    dims = tuple(None if item is None else integer(item) for item in shape[1:3])
    return Input((dims[0], dims[1]), None if shape[0] is None else integer(shape[0]))


def _operation(kind: str, config: JsonObject, name: str) -> Operation:
    """Parse one supported Keras layer and validate its execution options."""
    dtype = config.get("dtype", "float32")
    if isinstance(dtype, dict):
        dtype = object_value(dtype["config"])["name"]
    if dtype != "float32":
        raise ValueError(f"Only float32 layer policies are supported: {name}")
    if config.get("data_format", "channels_last") != "channels_last":
        raise ValueError("Only channels_last Keras graphs are supported")
    if kind == "InputLayer":
        return _input(config)
    if kind == "Conv2D":
        if pair(config.get("dilation_rate", [1, 1])) != (1, 1) or config.get("groups", 1) != 1:
            raise ValueError("Only ungrouped undilated RadImageNet convolutions are supported")
        return Conv(
            integer(config["filters"]),
            pair(config["kernel_size"]),
            pair(config["strides"]),
            Padding(string(config["padding"])),
            boolean(config["use_bias"]),
            Activation(string(config["activation"])),
        )
    if kind == "BatchNormalization":
        if config["axis"] not in (-1, 3) and config["axis"] != [3]:
            raise ValueError("Batch normalization must use the channel axis")
        return BatchNorm(
            number(config["epsilon"]), boolean(config["center"]), boolean(config["scale"])
        )
    if kind == "Activation":
        return Activation(string(config["activation"]))
    if kind in {"MaxPooling2D", "AveragePooling2D"}:
        return Pool(
            pair(config["pool_size"]),
            pair(config["strides"]),
            Padding(string(config["padding"])),
            kind == "AveragePooling2D",
        )
    if kind == "ZeroPadding2D":
        values = list_value(config["padding"])
        if len(values) != 2:
            raise ValueError("Expected two padding dimensions")
        return Pad((pair(values[0]), pair(values[1])))
    if kind in {"Add", "Concatenate"}:
        if kind == "Concatenate" and config["axis"] not in (-1, 3):
            raise ValueError("Concatenation must use the channel axis")
        return Merge(kind == "Concatenate")
    if kind in {"Lambda", "CustomScaleLayer"}:
        return _scale(kind, config, name)
    if kind in {"GlobalAveragePooling2D", "GlobalMaxPooling2D"}:
        return GlobalPool(kind == "GlobalAveragePooling2D", boolean(config.get("keepdims", False)))
    if kind == "Dense":
        return Dense(
            integer(config["units"]),
            boolean(config["use_bias"]),
            Activation(string(config["activation"])),
        )
    if kind == "Dropout":
        return Dropout()
    raise ValueError(f"Unsupported Keras layer {kind}: {name}")


def _scale(kind: str, config: JsonObject, name: str) -> ResidualScale:
    """Parse a permitted residual-scale layer without executing Lambda payloads."""
    if kind == "CustomScaleLayer":
        return ResidualScale(number(config["scale"]))
    # Old official InceptionResNetV2 embeds a Python lambda. Use only its
    # declared residual-block name and scalar argument, never executable code.
    block, _, index = name.rpartition("_")
    if block not in {"block35", "block17", "block8"} or not index.isdecimal():
        raise ValueError(f"Arbitrary Lambda layer is prohibited: {name}")
    expected = {"block35": 0.17, "block17": 0.1, "block8": 0.2}[block]
    if block == "block8" and index == "10":
        expected = 1.0
    scale = number(object_value(config["arguments"])["scale"])
    if scale != expected:
        raise ValueError(f"Unexpected official residual scale: {name}: {scale}")
    return ResidualScale(scale)


@dataclass(frozen=True)
class ArchitectureSpec:
    """Identity and limits of one family.

    ``anchor`` is a layer name that must exist and the four counts are the expected
    numbers of Conv, BatchNorm, Concatenate and residual-scale layers; together they
    identify the family. ``spatial`` is the last spatial tensor (``channels`` wide).
    ``input_size`` is the ``(height, width)`` default for a dynamic input and
    ``minimum_size`` the smallest accepted spatial dimension in pixels.
    """

    anchor: str
    spatial: str
    channels: int
    input_size: tuple[int, int]
    minimum_size: int
    convolutions: int
    normalizations: int
    concatenations: int
    residuals: int


ARCHITECTURES = {
    Architecture.RESNET50: ArchitectureSpec(
        "conv5_block3_out", "conv5_block3_out", 2048, (224, 224), 32, 53, 53, 0, 0
    ),
    Architecture.DENSENET121: ArchitectureSpec(
        "conv5_block16_concat", "relu", 1024, (224, 224), 32, 120, 121, 58, 0
    ),
    Architecture.INCEPTIONV3: ArchitectureSpec(
        "mixed10", "mixed10", 2048, (299, 299), 75, 94, 94, 15, 0
    ),
    Architecture.INCEPTIONRESNETV2: ArchitectureSpec(
        "conv_7b_ac", "conv_7b_ac", 1536, (299, 299), 75, 244, 204, 43, 40
    ),
}


def _family(nodes: tuple[Node, ...]) -> tuple[Architecture, str, int]:
    """Identify the supported architecture from its anchor and layer counts."""
    names = {node.name for node in nodes}
    counts = (
        sum(isinstance(n.operation, Conv) for n in nodes),
        sum(isinstance(n.operation, BatchNorm) for n in nodes),
        sum(isinstance(n.operation, Merge) and n.operation.concatenate for n in nodes),
        sum(isinstance(n.operation, ResidualScale) for n in nodes),
    )
    for family, spec in ARCHITECTURES.items():
        if spec.anchor in names and counts == (
            spec.convolutions,
            spec.normalizations,
            spec.concatenations,
            spec.residuals,
        ):
            return family, spec.spatial, spec.channels
    raise ValueError(
        "Expected a complete ResNet50, DenseNet121, InceptionV3 or InceptionResNetV2 graph"
    )


def parse_graph(value: JsonValue) -> Graph:
    """Parse a Keras ``model_config`` into a ``Graph`` of one supported family.

    Raises:
        ValueError: If the configuration is malformed, uses an unsupported layer or
            option, or does not match a complete ResNet50, DenseNet121, InceptionV3 or
            InceptionResNetV2 graph.
        KeyError: If a required configuration field is absent.
    """
    config = object_value(object_value(value)["config"])
    nodes: list[Node] = []
    seen: set[str] = set()
    for raw in list_value(config["layers"]):
        layer = object_value(raw)
        name = string(layer["name"])
        settings = object_value(layer["config"])
        operation = _operation(string(layer["class_name"]), settings, name)
        edges = tensor_references(layer.get("inbound_nodes", []))
        if name in seen or any(edge.source not in seen for edge in edges):
            raise ValueError(f"Duplicate layer or non-topological tensor edge: {name}")
        required = (
            0 if isinstance(operation, Input) else 2 if isinstance(operation, ResidualScale) else 1
        )
        if len(edges) < required or (not isinstance(operation, Merge) and len(edges) != required):
            raise ValueError(f"Invalid number of input tensor edges: {name}")
        nodes.append(Node(name, operation, edges))
        seen.add(name)
    inputs = [node.name for node in nodes if isinstance(node.operation, Input)]
    outputs = tensor_references(config["output_layers"])
    if len(inputs) != 1 or len(outputs) != 1 or outputs[0].source not in seen:
        raise ValueError("Exactly one input and output are required")
    reachable = {outputs[0].source}
    for node in reversed(nodes):
        if node.name in reachable:
            reachable.update(edge.source for edge in node.inputs)
    if reachable != seen:
        raise ValueError("Every declared graph node must contribute to the saved output")
    architecture, spatial, channels = _family(tuple(nodes))
    return Graph(architecture, tuple(nodes), inputs[0], outputs[0].source, spatial, channels)
