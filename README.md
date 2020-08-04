Dependencies:

* [protobuf](https://developers.google.com/protocol-buffers)
* [grpc](https://grpc.io/)

Build system:

* [meson](https://mesonbuild.com/)
* [ninja](https://ninja-build.org/)

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

Using Address Sanitizer:

https://mesonbuild.com/howtox.html#use-address-sanitizer

```
meson configure build -Db_sanitize=address
```
