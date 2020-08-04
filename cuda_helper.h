#pragma once

#include <stdexcept>

#include <absl/strings/str_format.h>
#include <cuda_runtime.h>

void cuda_check(const char *filename, int lineno, cudaError_t status) {
    if (status) {
        std::runtime_error(absl::StrFormat(
            "%s:%d cudaError_t(%d): %s", filename, lineno, status, cudaGetErrorString(status)));
    }
}

#define CUDA_CHECK(status) cuda_check(__FILE__, __LINE__, status)

template <class T, size_t size>
class gpu_array {
    T *p;
    constexpr size_t bytes() const {
        return sizeof(T) * size;
    }

public:
    gpu_array() {
        CUDA_CHECK(cudaMalloc(&p, bytes()));
    }
    ~gpu_array() {
        cudaFree(p);
    }
    gpu_array<T, size> &operator=(const std::array<T, size> &other) {
        CUDA_CHECK(cudaMemcpy(p, other.data(), bytes(), cudaMemcpyHostToDevice));
        return *this;
    }
    std::array<T, size> cpu() const {
        std::array<T, size> result;
        CUDA_CHECK(cudaMemcpy(result.data(), p, bytes(), cudaMemcpyDeviceToHost));
        return result;
    }
    void *data() {
        return p;
    }
};
