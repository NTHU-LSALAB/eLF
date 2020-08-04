#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

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
class ValueOperatorOp : public OpKernel {
public:
    explicit ValueOperatorOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<T>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output_flat = output_tensor->flat<T>();

        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        for (int i = 1; i < N; i++) {
            output_flat(i) = 0;
        }

        // Preserve the first input value if possible.
        if (N > 0)
            output_flat(0) = input(0);
    }
};

REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    ValueOperatorOp<float>);
REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    ValueOperatorOp<double>);
REGISTER_KERNEL_BUILDER(Name("ValueOperator").Device(DEVICE_GPU).TypeConstraint<int32_t>("T"),
    ValueOperatorOp<int32_t>);
