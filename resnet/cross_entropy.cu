#include <stdio.h>
#include "cross_entropy.h"
#define PAR(total, threads) <<<((total) + threads - 1) / threads, threads>>>

__global__ void nll_loss(float *loss, const float *logits_grad, const int *labels, int C,
                         int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < N) {
        int class_id = labels[index];
        loss[index] = -logits_grad[index * C + class_id];
    }
}

__global__ void nll_loss_backward(float *logits_grad, const float *loss,
                                  const int *labels, int C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < N * C) {
        logits_grad[index] = 0.0;
    }
    if(index < N) {
        auto loss_grad = -0.1 * loss[index] / N;
        int class_id = labels[index];
        logits_grad[index * C + class_id] = -loss_grad;
    }
}

void CrossEntropy::forward(float *loss, const float *act, const int *labels) {
    auto kAlgo = CUDNN_SOFTMAX_LOG;
    auto kMode = CUDNN_SOFTMAX_MODE_INSTANCE;
    float one = 1.0, zero = 0.0;
    auto logits = static_cast<float *>(global.get_workspace()) + 1;
    cudnnSoftmaxForward(global.cudnn_handle(), kAlgo, kMode, &one, dsc_io, act, &zero,
                        dsc_io, logits);
    nll_loss PAR(batch_size, 128)(loss, logits, labels, class_size, batch_size);
    // nll_loss
}

void CrossEntropy::backward(float *act_grad, const float *loss_grad, const int *labels) {
    auto kAlgo = CUDNN_SOFTMAX_LOG;
    auto kMode = CUDNN_SOFTMAX_MODE_INSTANCE;
    float one = 1.0, zero = 0.0;
    auto logits = static_cast<float *>(global.get_workspace());
    // nll_loss

    nll_loss_backward PAR(class_size * batch_size, 128)(logits, loss_grad, labels,
                                                        class_size, batch_size);
    auto st = cudnnSoftmaxBackward(global.cudnn_handle(), kAlgo, kMode, &one, dsc_io,
                                   global.get_workspace(), dsc_io, logits, &zero, dsc_io,
                                   act_grad);
}
