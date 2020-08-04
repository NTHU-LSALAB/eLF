#pragma once

#include <functional>

#include "communicator.h"

namespace elf {

// allreduce or broadcast
class Operator {
protected:
    Operator() {}

public:
    Operator(const Operator &) = delete;

    // execute the operator and return true if execution is scheduled successfully
    virtual bool execute_async(void *in,
        void *out,
        Communicator::DataType type,
        size_t count,
        std::function<void()> done_callback) = 0;
};

} // namespace elf