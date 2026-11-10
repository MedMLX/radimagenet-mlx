<div align="center">

# radimagenet-mlx

Extract radiology-image features with native MLX RadImageNet encoders: ResNet50, DenseNet121, InceptionV3 and InceptionResNetV2.

[![PyPI](https://img.shields.io/pypi/v/radimagenet-mlx)](https://pypi.org/project/radimagenet-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/radimagenet-mlx)](https://pypi.org/project/radimagenet-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/radimagenet-mlx/blob/v0.3.4/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/radimagenet-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/radimagenet-mlx/main/docs/showcase.png" alt="RadImageNet encoder features from radiology images" width="100%">

</div>

| Input | Output |
| :--- | :--- |
| 2D image | Pooled and spatial features |

| Model | Backbone | Weights download | Peak MLX memory |
| :--- | :--- | ---: | ---: |
| `radimagenet-densenet121` | DenseNet121 | 28 MB | 0.11 GB |
| `radimagenet-inceptionv3` | InceptionV3 | 87 MB | 0.18 GB |
| `radimagenet-resnet50` | ResNet50 | 94 MB | 0.19 GB |
| `radimagenet-inceptionresnetv2` | InceptionResNetV2 | 218 MB | 0.44 GB |

Each backbone downloads its own weights file. Peak memory is for a 1929×2207 chest radiograph.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "radimagenet-mlx>=0.3.4"
```

## Run

Each backbone has its own registered model ID:
`radimagenet-resnet50`, `radimagenet-densenet121`, `radimagenet-inceptionv3` and
`radimagenet-inceptionresnetv2`. Their runners take the same arguments:

```python
from radimagenet_mlx.medmlx import resnet50

result = resnet50(image_path="image.png", output_dir="outputs")
```

Only the selected backbone's weights download from `MedMLX/radimagenet-mlx` on first
use; `medmlx setup radimagenet-resnet50` downloads them ahead of time. For local weights,
pass a folder containing the backbone's weights file as `weights_path`.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | One Pillow-readable 2D image; a multi-frame file uses its first frame |
| `output_dir` | Directory for the outputs |
| `weights_path` | Optional folder containing the backbone's `.safetensors` file |
| `overwrite` | `False` by default, which refuses to replace existing outputs |
| `output_selection` | `"all"` (default), or `"pooled"` to write only `features.npy` and metadata |

| Output | File | Contents |
| --- | --- | --- |
| `features_npy` | `features.npy` | float32 `(1, C)` average-pooled vector |
| `spatial_features_npy` | `spatial_features.npy` | float32 `(1, H, W, C)` last spatial map |
| `features_json` | `features.json` | Run metadata, also returned as `details` |

The published checkpoints declare 224×224 inputs:

| Backbone | Weights file | Pooled width | Spatial map | Map size |
| :--- | :--- | ---: | :--- | :--- |
| ResNet50 | `RadImageNet-ResNet50_notop.safetensors` | 2048 | `conv5_block3_out` | 7×7 |
| DenseNet121 | `RadImageNet-DenseNet121_notop.safetensors` | 1024 | `relu` | 7×7 |
| InceptionV3 | `RadImageNet-InceptionV3_notop.safetensors` | 2048 | `mixed10` | 5×5 |
| InceptionResNetV2 | `RadImageNet-IRV2_notop.safetensors` | 1536 | `conv_7b_ac` | 5×5 |

- A checkpoint with a classifier head also writes `predictions.npy`; the published
  encoders have none.
- **Preprocessing** follows the upstream RadImageNet example: RGB, nearest-neighbor
  resize to the checkpoint's input size, Caffe-style BGR mean subtraction, then scaling
  by 1/255.
- Outputs that alias the input or each other are refused.
- Missing or unreadable images raise `InvalidInputError`; missing or mismatched weights
  raise `AssetNotReadyError`.

For a cohort, `run_batch` loads one model and streams actual model batches. Supply
one output folder per image; each folder keeps the single-image artifact format.
The iterator publishes completed images in input order and retains them if a later
image fails. Consume or close it to release the model.

```python
from radimagenet_mlx.medmlx import FeatureSession

with FeatureSession("RadImageNet-ResNet50_notop.safetensors") as session:
    for result in session.run_batch(
        image_paths=["slice1.png", "slice2.png"],
        output_dirs=["out/slice1", "out/slice2"],
        batch_size=2,
        output_selection="pooled",
    ):
        print(result["outputs"]["features_npy"])
```

A session can run further images or cohorts with the same loaded checkpoint.
`run_batch` from the same module accepts a checkpoint file as `weights_path` and
manages its own session. Fixed-batch checkpoints use their declared batch size;
it must fit `batch_size` and divide the cohort length.

### Local weights

Download one of the [native weights](https://huggingface.co/MedMLX/radimagenet-mlx) and pass the file:

```python
from huggingface_hub import hf_hub_download
from radimagenet_mlx.medmlx import run

weights = hf_hub_download(
    "MedMLX/radimagenet-mlx", "RadImageNet-ResNet50_notop.safetensors",
    revision="fd1c2663266360c899211c9a23ffc24f805d282a",
)
result = run(
    image_path="slice.png",
    weights_path=weights,
    output_dir="features",
)
print(result["outputs"]["features_npy"])
```

The safetensors file must contain float32 MLX-layout weights and declarative
`model_config` graph metadata. The published bundle contains four encoders without
classification heads. Classifier predictions require a checkpoint that includes a classifier.
`run` detects the backbone, or takes `backbone="resnet50"`, `"densenet121"`,
`"inceptionv3"` or `"inceptionresnetv2"`.

### Python API

```python
from pathlib import Path
from radimagenet_mlx import RadImageNetResNet50Mlx, prepare_image

encoder = RadImageNetResNet50Mlx(Path("RadImageNet-ResNet50_notop.safetensors"))
images = prepare_image(Path("slice.png"))  # float32 NHWC, shape (1, 224, 224, 3)
features = encoder(images)  # (1, 2048)
spatial = encoder.spatial_features(images)  # (1, 7, 7, 2048)
```

- `RadImageNetMlx(checkpoint_file)` detects the architecture; the family classes also
  enforce it.
- `extract(images)` returns spatial features, pooled features and any classifier
  predictions in one pass, and `predict` refuses checkpoints without a classifier.
- Encoders take prepared `BxHxWx3` arrays and do not check them for finite values.
- Checkpoints without a fixed input size default to 224×224 for ResNet50 and
  DenseNet121 and 299×299 for the Inception families (minimum 32 and 75).
- The loader reads graph metadata and float32 tensors without running Keras or
  serialized Python.
- Inference requires Apple Silicon with Metal.

## Use with MedMLX

Use the [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) to run `radimagenet-resnet50`, `radimagenet-densenet121`, `radimagenet-inceptionv3` and `radimagenet-inceptionresnetv2` from an MCP client. Weights download on first use.

```sh
uv tool install --python 3.13 "medmlx[vision,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe radimagenet-resnet50`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## License

[Apache-2.0](https://github.com/MedMLX/radimagenet-mlx/blob/v0.3.4/LICENSE). See [third-party notices](https://github.com/MedMLX/radimagenet-mlx/blob/v0.3.4/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.
