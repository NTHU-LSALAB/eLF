#include <catch2/catch.hpp>

#include <absl/strings/str_format.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "lkvs_impl.h"
#include "nccl_communicator.h"

#define CUDA_ASSERT(status)                                                                        \
    do {                                                                                           \
        if (status) {                                                                              \
            throw std::runtime_error(absl::StrFormat("%s:%d cudaError_t(%d): %s", __FILE__,        \
                __LINE__, status, cudaGetErrorString(status)));                                    \
        }                                                                                          \
    } while (0)

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
    {
        int device_count;
        CUDA_ASSERT(cudaGetDeviceCount(&device_count));
        if (device_count < 2) {
            FAIL("need at least 2 CUDA devices to run this test");
        }
    }

    LocalKeyValueStore lkvs;

    auto interfere1 = create_nccl_communicator(&lkvs, "other-var-1", 0, 1);
    auto interfere2 = create_nccl_communicator(&lkvs, "other-var-2", 0, 1);

#pragma omp parallel num_threads(2)
    {
        CUDA_ASSERT(cudaSetDevice(omp_get_thread_num()));
        auto comm = create_nccl_communicator(&lkvs, "var1", omp_get_thread_num(), 2);

        thrust::host_vector<TestType> H(4);
        if (omp_get_thread_num() == 0) {
            H[0] = 1;
            H[1] = 2;
            H[2] = 3;
            H[3] = 4;
        } else {
            H[0] = 0;
            H[1] = 8;
            H[2] = 3;
            H[3] = 6;
        }
        thrust::device_vector<TestType> Dsrc = H;
        thrust::device_vector<TestType> Ddst = H;

        comm->allreduce(thrust::raw_pointer_cast(Dsrc.data()),
            thrust::raw_pointer_cast(Ddst.data()), 4, Communicator::datatype_of<TestType>());
        H = Ddst;
#pragma omp critical
        {
            CHECK(H[0] == 1);
            CHECK(H[1] == 10);
            CHECK(H[2] == 6);
            CHECK(H[3] == 10);
        }

        comm->broadcast(thrust::raw_pointer_cast(Dsrc.data()),
            thrust::raw_pointer_cast(Ddst.data()), 0, 4, Communicator::datatype_of<TestType>());
        H = Ddst;
#pragma omp critical
        {
            CHECK(H[0] == 1);
            CHECK(H[1] == 2);
            CHECK(H[2] == 3);
            CHECK(H[3] == 4);
        }

        comm->broadcast(thrust::raw_pointer_cast(Dsrc.data()),
            thrust::raw_pointer_cast(Ddst.data()), 1, 4, Communicator::datatype_of<TestType>());
        H = Ddst;
#pragma omp critical
        {
            CHECK(H[0] == 0);
            CHECK(H[1] == 8);
            CHECK(H[2] == 3);
            CHECK(H[3] == 6);
        }
    }
}
