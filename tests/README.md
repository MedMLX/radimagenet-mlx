The Keras preprocessing, runner, and full-graph parity tests use a chest X-ray
at `~/.cache/medmlx/test-data/chest_xray.jpg`. The recorder writes image-derived
inputs and reference outputs to `work/real-reference`, outside Git.

Run `scripts/record_real_reference.py` in an environment with TensorFlow
2.20.0, Keras 3.15.0, NumPy, and Pillow. Pass `--xray` to select a different
chest X-ray. After recording, run `python -m pytest tests/test_preprocessing.py
tests/test_medmlx.py tests/test_applications.py tests/test_equivalence.py` in
the project environment. The checked-in application fixtures retain only
model configuration and seeded-weight recipes.
