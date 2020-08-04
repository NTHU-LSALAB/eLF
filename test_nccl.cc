#include <catch2/catch.hpp>

#include <algorithm>
#include <array>

#include <absl/strings/str_format.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "cuda_helper.h"
#include "lkvs_impl.h"
#include "nccl_communicator.h"

TEMPLATE_TEST_CASE("nccl communicator",
    "[communicator][nccl]",
    int8_t,
    int32_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    float,
    double) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        WARN("Less than 2 CUDA devices. This test may fail.");
    }

    elf::LocalKeyValueStore lkvs;

    auto interfere1 = elf::create_nccl_communicator(&lkvs, "other-var-1", 0, 1);
    auto interfere2 = elf::create_nccl_communicator(&lkvs, "other-var-2", 0, 1);

#pragma omp parallel num_threads(2)
    {
        CUDA_CHECK(cudaSetDevice(std::min(omp_get_thread_num(), device_count - 1)));
        auto comm = create_nccl_communicator(&lkvs, "var1", omp_get_thread_num(), 2);

        std::array<TestType, 4> H;
        if (omp_get_thread_num() == 0) {
            H = {1, 2, 3, 4};
        } else {
            H = {0, 8, 3, 6};
        }
        gpu_array<TestType, 4> Dsrc, Ddst;
        Dsrc = H;

        comm->allreduce(Dsrc.data(), Ddst.data(), 4, elf::Communicator::datatype_of<TestType>());
#pragma omp critical
        { CHECK(Ddst.cpu() == std::array<TestType, 4>{1, 10, 6, 10}); }

        comm->broadcast(Dsrc.data(), Ddst.data(), 0, 4, elf::Communicator::datatype_of<TestType>());
#pragma omp critical
        { CHECK(Ddst.cpu() == std::array<TestType, 4>{1, 2, 3, 4}); }

        comm->broadcast(Dsrc.data(), Ddst.data(), 1, 4, elf::Communicator::datatype_of<TestType>());
#pragma omp critical
        { CHECK(Ddst.cpu() == std::array<TestType, 4>{0, 8, 3, 6}); }
    }
}
