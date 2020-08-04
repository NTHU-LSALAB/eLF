Prerequisites:

* [clang](https://clang.llvm.org/)
* [protobuf](https://developers.google.com/protocol-buffers)
* [grpc](https://grpc.io/)
* [ninja](https://ninja-build.org/)
* [meson](https://mesonbuild.com/) >= 0.55

Build instructions:

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
