# Third-party notices

MedMLX-authored additions are licensed under [Apache-2.0](LICENSE). Upstream
materials retain the terms below.

## RadImageNet

This package implements native MLX inference for the encoders and preprocessing
published by [RadImageNet](https://github.com/BMEII-AI/RadImageNet). Its upstream
[MIT license](https://github.com/BMEII-AI/RadImageNet/blob/main/LICENSE) follows:

```text
MIT License

Copyright (c) 2021 BMEII-AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Keras reference implementations

Keras application architectures and preprocessing provide the independent
numerical references for the tests.
The pinned Keras source retains its
[Apache-2.0 license](https://github.com/keras-team/keras/blob/9d1bbf9aae730708cc6e0d7104860c1605c0e31b/LICENSE).

Pretrained HDF5 checkpoints and medical images are separate assets whose terms
are set by their publishers. Dependencies declared in
[pyproject.toml](pyproject.toml) retain their respective licenses.

## Showcase image

The CT images are from [AMOS22](https://doi.org/10.5281/zenodo.7262581),
Ji, Yuanfeng et al. (2022). The Zenodo record lists
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), while the
dataset's own metadata lists
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The showcase uses cropped views and cosine-similarity retrieval scores and is
shared under CC BY-SA 4.0 pending clarification of the source terms.
