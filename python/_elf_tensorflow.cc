#include <sstream>
#include <string>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "operator.h"

using namespace tensorflow;

void *stringToPtr(const std::string &s) {
    std::istringstream iss(s);
    uintptr_t p;
    iss >> p;
    return (void *)p;
}

REGISTER_OP("ValueOperator")
    .Attr("T: {float32, float64, int32}")
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
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
        auto tensor = context->input(0);
        Tensor *output;
        OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, tensor.shape(), &output), done);
        auto in = tensor.tensor_data().data();
        auto out = output->tensor_data().data();
        elf_op->execute_async(in, out, tensor.NumElements(), elf::Communicator::datatype_of<T>(), done);
    }

private:
    elf::Operator *elf_op;
};

REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    ValueOperatorOp<float>);
REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    ValueOperatorOp<double>);
REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<int32_t>("T"),
    ValueOperatorOp<int32_t>);
