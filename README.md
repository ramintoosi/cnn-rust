# Train a CNN in Rust
This repository contains a CNN training and inference code in Rust using `tch-rust` crate.

## Blog post
You can find the description of this code in my [blog post](https://ramintoosi.ir/posts/2024/06/blog-post-1/).

## How to build

We need to build `tch-rust`. For a full description check [tch-rust](https://github.com/LaurentMazare/tch-rs).

### Python
create a python env and install `typing-extensions` and `pyyaml`.

### Download and build libtorch

```shell
git clone -b v2.3.0 --recurse-submodule https://github.com/pytorch/pytorch.git --depth 1
cd pytorch
USE_CUDA=ON BUILD_SHARED_LIBS=ON python setup.py build
```

### Setup envs
```shell
export LIBTORCH=/path/to/pytorch/build/lib.linux-x86_64-cpython-310/torch/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/path/to/pytorch//build/lib.linux-x86_64-cpython-310/torch/lib
```
Change ```lib.linux-x86_64-cpython-310``` accordingly.

**Note:** You can download libtorch pre-build files from Pytorch site.

### Download and build the project

```shell
git clone https://github.com/ramintoosi/cnn-rust
cargo build --release
```