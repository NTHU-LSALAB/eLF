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
    virtual ~Operator() {}

    // execute the operator and return true if execution is scheduled successfully
    virtual bool execute_async(void *in,
        void *out,
        size_t count,
        Communicator::DataType type,
        std::function<void()> done_callback) = 0;
};

} // namespace elf
