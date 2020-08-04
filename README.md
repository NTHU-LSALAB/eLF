Dependencies:

* [abseil-cpp](https://github.com/abseil/abseil-cpp)
* [protocol buffers](https://developers.google.com/protocol-buffers)
* [grpc](https://grpc.io/)
* [catch2](https://github.com/catchorg/Catch2) (for testing)

Build system:

* [meson](https://mesonbuild.com/)
* [ninja](https://ninja-build.org/)
* [cmake](https://cmake.org/)

Build instructions:

1.  Configure the build:

    ```
    meson build
    ```

    This command configures the build in a directory named `build`.

    It is recommended that you use `clang++` as the compiler and `lld` as the linker. To do so, use the following command instead:

    ```
    CXX=clang++ CXX_LD=ldd meson build
    ```

2.  Build eLF:

    ```
    ninja -C build
    ```

3.  Do some tests

    ```
    meson test -C build -v
    ```
