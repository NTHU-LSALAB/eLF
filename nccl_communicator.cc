#include <iostream>
#include <stdexcept>
#include <string>

#include <nccl.h>

#include <absl/strings/str_format.h>

#include "communicator.h"
#include "kvs.h"
#include "nccl_communicator.h"

namespace elf {

void nccl_assert(const char *filename, int lineno, ncclResult_t result) {
    if (result) {
        throw std::runtime_error(absl::StrFormat(
            "%s:%d: ncclResult_t(%d): %s", filename, lineno, result, ncclGetErrorString(result)));
    }
}

#define NCCL_ASSERT(expr) nccl_assert(__FILE__, __LINE__, expr)

ncclDataType_t comm_type_to_nccl(Communicator::DataType type) {
    switch (type) {
    default:
        throw std::runtime_error(absl::StrFormat("invalid type %d", type));
    case Communicator::i8:
        return ncclInt8;
    case Communicator::i32:
        return ncclInt32;
    case Communicator::i64:
        return ncclInt64;
    case Communicator::u8:
        return ncclUint8;
    case Communicator::u32:
        return ncclUint32;
    case Communicator::u64:
        return ncclUint64;
    case Communicator::f32:
        return ncclFloat32;
    case Communicator::f64:
        return ncclFloat64;
    }
}

class NcclCommunicator : public Communicator {
    KeyValueStore *kvs;
    const std::string identifier;
    const int rank;
    const int size;
    ncclComm_t comm;

public:
    NcclCommunicator(KeyValueStore *kvs, const std::string &identifier, int rank, int size)
        : kvs(kvs), identifier(identifier), rank(rank), size(size) {
        init();
    }
    ~NcclCommunicator() {
        ncclCommDestroy(comm);
    }

    void allreduce(void *src, void *dst, size_t count, Communicator::DataType datatype) override {
        NCCL_ASSERT(ncclAllReduce(src, dst, count, comm_type_to_nccl(datatype), ncclSum, comm, 0));
    }

    void broadcast(void *src,
        void *dst,
        int root,
        size_t count,
        Communicator::DataType datatype) override {
        NCCL_ASSERT(ncclBroadcast(src, dst, count, comm_type_to_nccl(datatype), root, comm, 0));
    }

private:
    void init() {
        ncclUniqueId nccl_id;
        if (rank == 0) {
            NCCL_ASSERT(ncclGetUniqueId(&nccl_id));
            kvs->set(identifier, std::string(nccl_id.internal, NCCL_UNIQUE_ID_BYTES));
        } else {
            std::string id_str = kvs->get(identifier).get();
            memcpy(nccl_id.internal, id_str.c_str(), NCCL_UNIQUE_ID_BYTES);
        }
        NCCL_ASSERT(ncclCommInitRank(&comm, size, nccl_id, rank));
    }
};

std::unique_ptr<Communicator>
create_nccl_communicator(KeyValueStore *kvs, const std::string &identifier, int rank, int size) {
    return std::make_unique<NcclCommunicator>(kvs, identifier, rank, size);
}

} // namespace elf
