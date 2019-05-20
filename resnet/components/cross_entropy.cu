#include <stdio.h>
#include "../helper/global.h"
#include "cross_entropy.h"
#define PAR(total, threads) <<<((total) + threads - 1) / threads, threads>>>

// __global__ void nll_loss(float *loss, const float *logits_grad, const int *labels, int C,
//                          int N) {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;
//     if(index < N) {
//         int class_id = labels[index];
//         loss[index] = -logits_grad[index * C + class_id];
//     }
// }

// __global__ void nll_loss_backward(float *logits_grad, float rate, const float *loss,
//                                   const int *labels, int C, int N) {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;
//     if(index < N * C) {
//         logits_grad[index] = 0.0;
//     }
//     if(index < N) {
//         auto loss_grad = -rate * loss[index] / N;
//         int class_id = labels[index];
//         logits_grad[index * C + class_id] = -loss_grad;
//     }
// }

#define CUDA_1D_KERNEL_LOOP(i, n)                                  \
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

__global__ void LabelCrossEntropyKernel(const int N, const int D, const float *Xdata,
                                        const int *labeldata, const float log_threshold,
                                        float *Ydata) {
    CUDA_1D_KERNEL_LOOP(i, N) {
        Ydata[i] = -logf(fmaxf(Xdata[i * D + labeldata[i]], log_threshold));
    }
}
__global__ void LabelCrossEntropyGradientKernel(const int N, const int D, float rate,
                                                const float *Xdata, const int *labeldata,
                                                const float *dYdata,
                                                const float log_threshold,
                                                float *dXdata) {
    CUDA_1D_KERNEL_LOOP(i, N) {
        int idx = i * D + labeldata[i];
        dXdata[idx] = -dYdata[i] / fmaxf(Xdata[idx], log_threshold) * -rate / N;
    }
}

void CrossEntropy::forward(float *loss, const float *act, const int *labels) {
    // auto kAlgo = CUDNN_SOFTMAX_LOG;
    // auto kMode = CUDNN_SOFTMAX_MODE_INSTANCE;
    // float one = 1.0, zero = 0.0;
    // auto logits = static_cast<float *>(global.get_workspace()) + 1;
    // cudnnSoftmaxForward(global.cudnn_handle(), kAlgo, kMode, &one, dsc_io, act, &zero,
    //                     dsc_io, logits);
    // nll_loss PAR(batch_size, 128)(loss, logits, labels, class_size, batch_size);
    // nll_loss
    LabelCrossEntropyKernel PAR(batch_size, 128)(batch_size, class_size, act, labels, 1e-20, loss);
}

void CrossEntropy::backward(float *act_grad, float rate, const float* act, const float *loss_grad,
                            const int *labels) {
    // auto kAlgo = CUDNN_SOFTMAX_LOG;
    // auto kMode = CUDNN_SOFTMAX_MODE_INSTANCE;
    // float one = 1.0, zero = 0.0;
    // auto logits = static_cast<float *>(global.get_workspace());
    // // nll_loss

    // nll_loss_backward PAR(class_size * batch_size, 128)(logits, rate, loss_grad, labels,
    //                                                     class_size, batch_size);
    // auto st = cudnnSoftmaxBackward(global.cudnn_handle(), kAlgo, kMode, &one, dsc_io,
    //                                global.get_workspace(), dsc_io, logits, &zero, dsc_io,
    //                                act_grad);
    LabelCrossEntropyGradientKernel PAR(batch_size, 128)(batch_size, class_size, rate, act, labels, loss_grad, 1e-20, act_grad); 
}
