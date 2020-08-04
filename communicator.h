#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

class Communicator {

protected:
    Communicator() {}

public:
    enum DataType { invalid, i8, i32, i64, u8, u32, u64, f32, f64 };

    Communicator(const Communicator &) = delete;
    virtual ~Communicator() {}
    virtual void allreduce(void *src, void *dst, size_t count, Communicator::DataType datatype) = 0;
    virtual void
    broadcast(void *src, void *dst, int root, size_t count, Communicator::DataType datatype) = 0;

    template <typename T>
    static constexpr DataType datatype_of() {
        if (std::is_same<T, int8_t>::value) {
            return i8;
        }
        if (std::is_same<T, int32_t>::value) {
            return i32;
        }
        if (std::is_same<T, int64_t>::value) {
            return i64;
        }
        if (std::is_same<T, uint8_t>::value) {
            return u8;
        }
        if (std::is_same<T, uint32_t>::value) {
            return u32;
        }
        if (std::is_same<T, uint64_t>::value) {
            return u64;
        }
        if (std::is_same<T, float>::value) {
            return f32;
        }
        if (std::is_same<T, double>::value) {
            return f64;
        }
        return invalid;
    }
};
