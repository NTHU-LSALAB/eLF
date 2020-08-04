#include <sstream>
#include <string>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_util.h>

#include "operator.h"

using namespace tensorflow;

void *stringToPtr(const std::string &s) {
    auto result = std::stoul(s, 0, 0);
    static_assert(
        std::is_same<uintptr_t, decltype(result)>::value, "stoul cannot represent a pointer");
    return (void *)result;
}

REGISTER_OP("ValueOperator")
    .Attr("T: {float32, float64, int32, int64}")
    .Attr("handle: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

template <class T>
class ValueOperatorOp : public AsyncOpKernel {
public:
    explicit ValueOperatorOp(OpKernelConstruction *context) : AsyncOpKernel(context) {
        std::string handle;
        OP_REQUIRES_OK(context, context->GetAttr("handle", &handle));
        elf_op = (elf::Operator *)stringToPtr(handle);
        OP_REQUIRES(context, elf_op != nullptr, errors::InvalidArgument("handle cannot be 0"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
        auto tensor = context->input(0);
        Tensor *output;
        OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, tensor.shape(), &output), done);
        auto in = tensor.flat<T>().data();
        auto out = output->flat<T>().data();
        elf_op->execute_async(
            in, out, tensor.NumElements(), elf::Communicator::datatype_of<T>(), done);
    }

private:
    elf::Operator *elf_op;
};

// CPU variables like the step counter is (maybe) OK to ignore
class NoopForCpuOp : public OpKernel {
public:
    explicit NoopForCpuOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        auto input = context->input(0);
        Tensor *output;
        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
        *output = tensor::DeepCopy(input);

        std::cerr << input.DebugString() << "\n";
    }
};

REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    ValueOperatorOp<float>);
REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    ValueOperatorOp<double>);
REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<int32_t>("T"),
    ValueOperatorOp<int32_t>);
REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<int64>("T"),
    ValueOperatorOp<int64>);

REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_CPU), NoopForCpuOp);
