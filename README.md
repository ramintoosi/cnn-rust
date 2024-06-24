### python
create a python env and install typing-extensions and pyyaml
### Download and build libtorch

```bash
git clone -b v2.3.0 --recurse-submodule https://github.com/pytorch/pytorch.git --depth 1
cd pytorch
USE_CUDA=ON BUILD_SHARED_LIBS=ON python setup.py build
```

### setup envs
```bash
export LIBTORCH=/path/to/pytorch/build/lib.linux-x86_64-cpython-310/torch/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/home/phantom/RustProjects/torchtest/pytorch/build/lib.linux-x86_64-cpython-310/torch/lib
```
Change ```lib.linux-x86_64-cpython-310``` accordingly
