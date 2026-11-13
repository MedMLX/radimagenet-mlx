"""Record pinned Keras image and graph outputs from a chest X-ray."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras.src.applications.inception_resnet_v2 import CustomScaleLayer
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def seed_parameters(model, specs, seed, normalization_seed=None):
    """Populate Keras layers using the existing independent seeded-weight recipe."""
    rng = np.random.Generator(np.random.PCG64(seed))
    layers = {}
    normalization_rng = (
        np.random.default_rng(normalization_seed) if normalization_seed is not None else None
    )
    for spec in specs:
        shape, key = spec["shape"], spec["key"]
        if normalization_rng is not None and key in {
            "gamma:0",
            "beta:0",
            "moving_mean:0",
            "moving_variance:0",
        }:
            low, high = (-0.2, 0.2) if key in {"beta:0", "moving_mean:0"} else (0.5, 1.5)
            values = normalization_rng.uniform(low, high, shape)
        elif key == "kernel:0":
            values = rng.normal(0.0, 1.0 / np.sqrt(np.prod(shape[:-1])), shape)
        elif key == "gamma:0":
            values = rng.uniform(0.8, 1.2, shape)
        elif key == "moving_variance:0":
            values = rng.uniform(0.7, 1.3, shape)
        else:
            values = rng.normal(0.0, 0.02, shape)
        layers.setdefault(spec["layer"], []).append(values.astype(np.float32))
    for name, values in layers.items():
        model.get_layer(name).set_weights(values)


def main() -> None:
    """Use the pinned Keras directory iterator and source graph on real frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xray", type=Path, default=Path.home() / ".cache/medmlx/test-data/chest_xray.jpg"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work/real-reference")
    args = parser.parse_args()
    if tf.__version__ != "2.20.0" or keras.__version__ != "3.15.0":
        raise ValueError("TensorFlow 2.20.0 and Keras 3.15.0 are required")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(FIXTURES / "keras_checkpoint.npz", allow_pickle=False) as checkpoint:
        metadata = json.loads(checkpoint["metadata"].item())
    metadata["image_source"] = "chest X-ray"
    metadata.pop("input_seed", None)
    metadata.pop("slice_input_seed", None)
    shapes = {
        "rgb": (29, 17),
        "gray": (37, 23),
        "rgba": (19, 13),
        "gray16": (7, 11),
        "rgb_same": (224, 224),
        "rgb_downsample": (266, 238),
        "slice_gray": (37, 23),
        "slice_rgba": (13, 9),
        "slice_gray16": (30, 40),
    }
    outputs = {"metadata": np.asarray(json.dumps(metadata)), "cases": np.asarray(list(shapes))}
    with (
        Image.open(args.xray) as source,
        tempfile.TemporaryDirectory(prefix="radimagenet-xray-") as temp,
    ):
        root = Path(temp)
        for case, size in shapes.items():
            gray = source.convert("L").resize(size, Image.Resampling.LANCZOS)
            if "gray16" in case:
                image = Image.fromarray(np.asarray(gray, dtype=np.uint16) * np.uint16(257))
            elif "rgba" in case:
                image = gray.convert("RGBA")
            elif "rgb" in case:
                image = gray.convert("RGB")
            else:
                image = gray
            directory = root / case / "class"
            directory.mkdir(parents=True)
            image.save(directory / "image.png")
            outputs[f"png::{case}"] = np.frombuffer(
                (directory / "image.png").read_bytes(), dtype=np.uint8
            )
            iterator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=keras.applications.imagenet_utils.preprocess_input,
                rescale=1.0 / 255.0,
            ).flow_from_directory(
                str(directory.parent),
                target_size=(224, 224),
                batch_size=1,
                class_mode=None,
                shuffle=False,
                interpolation="nearest",
            )
            outputs[f"prepared::{case}"] = next(iterator)
    np.savez_compressed(args.output / "keras_preprocessing.npz", **outputs)
    with np.load(FIXTURES / "keras_checkpoint.npz", allow_pickle=False) as checkpoint:
        config = json.loads(checkpoint["model_config"].item())
        specs = json.loads(checkpoint["weight_specs"].item())
        model = keras.Model.from_config(config["config"])
        seed_parameters(model, specs, int(checkpoint["weight_seed"].item()))
    for case, output in (
        ("rgb", "keras_resnet50_runner"),
        ("slice_gray", "keras_resnet50_runner_gray"),
    ):
        images = outputs[f"prepared::{case}"]
        spatial = model(images, training=False).numpy()
        pooled = np.mean(spatial, axis=(1, 2), dtype=np.float32)
        recorded_metadata = dict(metadata)
        np.savez_compressed(
            args.output / f"{output}.npz",
            metadata=json.dumps(recorded_metadata),
            images=images,
            spatial=spatial,
            pooled=pooled,
        )

    # The remaining full-graph references use the same source photograph at the
    # model-specific spatial sizes. All image-derived arrays stay outside Git.
    sizes = {
        "resnet50": (35, 49),
        "densenet121": (64, 80),
        "inceptionv3": (75, 91),
        "inceptionresnetv2": (75, 91),
    }
    with Image.open(args.xray) as source:
        for family, (height, width) in sizes.items():
            frame = np.asarray(source.convert("RGB").resize((width, height)), dtype=np.float32)
            images = frame[None] / np.float32(255.0) - np.float32(0.5)
            for suffix in ("spatial", "classifier"):
                name = f"application_{family}_{suffix}.npz"
                with np.load(FIXTURES / name, allow_pickle=False) as archive:
                    saved = {
                        key: archive[key]
                        for key in archive.files
                        if key not in {"images", "expected"}
                    }
                config = json.loads(saved["model_config"].item())
                model = keras.Model.from_config(
                    config["config"], custom_objects={"CustomScaleLayer": CustomScaleLayer}
                )
                seed_parameters(
                    model, json.loads(saved["weight_specs"].item()), int(saved["seed"].item())
                )
                expected = model(images, training=False).numpy()
                metadata = json.loads(saved["metadata"].item())
                metadata["image_source"] = "chest X-ray"
                metadata.pop("input_seed", None)
                saved["metadata"] = np.asarray(json.dumps(metadata))
                np.savez_compressed(args.output / name, **saved, images=images, expected=expected)
                del model
                keras.backend.clear_session()
                gc.collect()
        for case, (batch, height, width), checkpoint_name in (
            ("square", (2, 32, 32), "keras_checkpoint"),
            ("small_rectangular", (1, 35, 49), "keras_checkpoint"),
            ("rectangular", (2, 35, 49), "keras_checkpoint"),
            ("initialized", (2, 64, 64), "keras_initialized_checkpoint"),
        ):
            frame = np.asarray(
                source.convert("RGB").resize((width, height)), dtype=np.float32
            ) / np.float32(255.0)
            if case == "rectangular":
                frame -= np.float32(0.5)
            images = np.stack([frame if index == 0 else frame[::-1] for index in range(batch)])
            with np.load(FIXTURES / f"{checkpoint_name}.npz", allow_pickle=False) as checkpoint:
                config = json.loads(checkpoint["model_config"].item())
                model = keras.Model.from_config(config["config"])
                normalization_seed = (
                    int(checkpoint["normalization_seed"].item())
                    if "normalization_seed" in checkpoint
                    else None
                )
                seed_parameters(
                    model,
                    json.loads(checkpoint["weight_specs"].item()),
                    int(checkpoint["weight_seed"].item()),
                    normalization_seed,
                )
                metadata = json.loads(checkpoint["metadata"].item())
            metadata["image_source"] = "chest X-ray"
            metadata.pop("input_seed", None)
            metadata.pop("slice_input_seed", None)
            spatial = model(images, training=False).numpy()
            pooled = np.mean(spatial, axis=(1, 2), dtype=np.float32)
            np.savez_compressed(
                args.output / f"keras_resnet50_{case}.npz",
                metadata=json.dumps(metadata),
                images=images,
                spatial=spatial,
                pooled=pooled,
            )
            del model
            keras.backend.clear_session()
            gc.collect()


if __name__ == "__main__":
    main()
