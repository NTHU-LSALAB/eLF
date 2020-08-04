## Prerequisites

Please have these installed before proceeding

* [Clang](https://clang.llvm.org/)
* [Protocol Buffers](https://developers.google.com/protocol-buffers)
* [gRPC](https://grpc.io/)
* [Ninja](https://ninja-build.org/)
* [Meson](https://mesonbuild.com/) >= 0.55
* [TensorFlow](https://www.tensorflow.org/) < 2
* CUDA, [NCCL](https://developer.nvidia.com/nccl)

## Build instructions

1.  Configure the build:

    ```
    CXX=clang++ meson build
    ```

    This command configures the build in a directory named `build`.

    It is recommended that you use `lld` as the linker. To do so, use the following command instead:

    ```
    CXX=clang++ CXX_LD=lld meson build
    ```

2.  Build eLF:

    ```
    ninja -C build
    ```

3.  Do some tests

    ```
    meson test -C build -v
    ```


## Running

See `examples/`
