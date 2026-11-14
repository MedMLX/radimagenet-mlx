"""Native arrays and serialized contracts used by the RadImageNet package."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict

import numpy as np
from medmlx_core.outputs import Artifact
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx.core import array

type HostArray = NDArray[np.generic]
type FloatArray = NDArray[np.float32]
type RealArray = NDArray[np.bool_ | np.integer | np.floating]
type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]


class ArrayRuntime(Protocol):
    def eval(self, *arrays: array) -> None:
        """Evaluate the supplied MLX arrays."""
        ...


class LayerConfig(TypedDict):
    epsilon: NotRequired[float]


class Layer(TypedDict):
    class_name: str
    name: str
    config: LayerConfig


class NetworkConfig(TypedDict):
    layers: list[Layer]
    output_layers: list[str | list[str | int]]


class ModelConfig(TypedDict):
    config: NetworkConfig


class WeightSpec(TypedDict):
    layer: str
    key: str
    shape: list[int]


class FeatureMetadata(TypedDict):
    """Metadata written to ``features.json`` and returned as ``details`` (schema 3).

    ``shape`` is the pooled feature shape ``(B, C)`` and ``spatial_shape`` the spatial
    map shape ``(B, H, W, C)`` when requested. Pooled-only runs record
    ``output_selection="pooled"`` and omit spatial/classifier output shapes.
    ``model_path`` is the absolute POSIX checkpoint path.
    ``preprocessing`` is ``"upstream"`` (Caffe-mode mean subtraction, then 1/255) and
    ``interpolation`` the resize method (``"nearest"``).
    """

    schema_version: int
    shape: list[int]
    dtype: str
    model_path: str
    weights_revision: str | None
    backbone: str
    classifier_output: NotRequired[str]
    classification_shape: NotRequired[list[int]]
    feature_layer: str
    spatial_feature_layer: str
    spatial_shape: NotRequired[list[int]]
    output_selection: NotRequired[str]
    preprocessing: str
    interpolation: str
    backend: str


class FeatureOutputs(TypedDict):
    features: str
    metadata: str
    spatial_features: NotRequired[str]
    predictions: NotRequired[str]


class RunResult(TypedDict):
    outputs: FeatureOutputs
    artifacts: dict[str, Artifact]
    details: FeatureMetadata
