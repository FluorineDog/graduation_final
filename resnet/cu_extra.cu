#include <math.h>
#include <cudnn.h>
#include "global.h"
#include "descriptor.h"
#include "../../../../../usr/include/cudnn.h"
#include "../../../../../usr/local/cuda/include/device_launch_parameters.h"

//__global__ void add_biased_impl(float *out, int len, int total_size, float alpha, float *in) {
//    auto id = threadIdx.x + blockIdx.x * blockDim.x;
//    if (id < total_size) {
//        auto k_id = id % len;
//        out[id] = alpha * in[k_id];
//    }
//}
//
//void add_biased(float *out, int len, int batch, float alpha, float *in) {
//    int size = batch * len;
//    add_biased_impl << < (size + 255) / 256, 256 >> > (out, len, size, alpha, in);
//}
#define PAR(total, threads) <<<((total) + threads - 1) / threads, threads>>>


__global__
void nll_loss(float *loss, const float *logits_grad, const int *labels, int C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        int class_id = labels[index];
        atomicAdd(loss, logits_grad[index * C + class_id]);
    }
}

__global__
void nll_loss_backward(float* logits_grad, float final, const int* labels, int C, int N){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N * C) {
        logits_grad[index] = 0.0;
    }
    if(index < N) {
        int class_id = labels[index];
        logits_grad[index * C + class_id] = final;
    }
}


class CrossEntropyFunctor {
public:
    CrossEntropyFunctor(int class_size, int batch_size) : class_size(class_size), batch_size(batch_size),
                                                          dsc_io({batch_size, class_size, 1, 1}) {
    }

    void forward(float *loss, const float *act, const int *labels) {
        auto kAlgo = CUDNN_SOFTMAX_LOG;
        auto kMode = CUDNN_SOFTMAX_MODE_INSTANCE;
        float one = 1.0, zero = 0.0;
        auto logits = (float *) global.get_workspace();
        cudnnSoftmaxForward(global.cudnn_handle(), kAlgo, kMode, &one, dsc_io, act, &zero, dsc_io, logits);
        nll_loss PAR(batch_size, 128)(loss, logits, labels, class_size, batch_size);
        // nll_loss
    }

    void backward(float *act_grad, float loss_grad, const int *labels) {
        auto kAlgo = CUDNN_SOFTMAX_LOG;
        auto kMode = CUDNN_SOFTMAX_MODE_INSTANCE;
        float one = 1.0, zero = 0.0;
        auto logits = (float *) global.get_workspace();
        // nll_loss
        nll_loss_backward PAR(class_size*batch_size, 128)(logits, loss_grad, labels, class_size, batch_size);
        cudnnSoftmaxBackward(global.cudnn_handle(), kAlgo, kMode, &one, dsc_io, global.get_workspace(), dsc_io,
                             logits, &zero, dsc_io, act_grad);

    }

    size_t workspace() {
        return class_size * batch_size * sizeof(float);
    }

private:
    int class_size;
    int batch_size;
    TensorDescriptor dsc_io;
};

