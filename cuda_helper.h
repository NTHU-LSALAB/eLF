#pragma once

#include <stdexcept>

#include <absl/strings/str_format.h>
#include <cuda_runtime.h>

#define CUDA_ASSERT(status)                                                                        \
    do {                                                                                           \
        if (status) {                                                                              \
            throw std::runtime_error(absl::StrFormat("%s:%d cudaError_t(%d): %s", __FILE__,        \
                __LINE__, status, cudaGetErrorString(status)));                                    \
        }                                                                                          \
    } while (0)

template <class T, size_t size>
class gpu_array {
    T *p;
    constexpr size_t bytes() const {
        return sizeof(T) * size;
    }

public:
    gpu_array() {
        CUDA_ASSERT(cudaMalloc(&p, bytes()));
    }
    ~gpu_array() {
        cudaFree(p);
    }
    gpu_array<T, size> &operator=(const std::array<T, size> &other) {
        CUDA_ASSERT(cudaMemcpy(p, other.data(), bytes(), cudaMemcpyHostToDevice));
        return *this;
    }
    std::array<T, size> cpu() const {
        std::array<T, size> result;
        CUDA_ASSERT(cudaMemcpy(result.data(), p, bytes(), cudaMemcpyDeviceToHost));
        return result;
    }
    void *data() {
        return p;
    }
};
