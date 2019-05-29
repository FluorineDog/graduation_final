#pragma once
#include "../helper/common.h"
#include "descriptor.h"

class PoolingFunctor {
  public:
    PoolingFunctor(int H, int W, int padding, int stride)
        : dsc_pool(H, W, padding, stride) {
            
    }
    void forward(float* out, const float* in) {
        float alpha = 1.0;
        float beta = 0.0;
        cudnnPoolingForward(global.cudnn_handle(), dsc_pool, &alpha, dsc_in, in, &beta,
                            dsc_out, out);
    }
    dim_t dims_out() {
        return dsc_out.dims();
    }
    void backward(float* in_grad, const float* in, const float* out_grad,
                  const float* out) {
        float alpha = 1.0;
        float beta = 1.0;
        cudnnPoolingBackward(global.cudnn_handle(), dsc_pool, &alpha, dsc_out, out,
                             dsc_out, out_grad, dsc_in, in, &beta, dsc_in, in_grad);
    }

  private:
    PoolingDescriptor dsc_pool;
    TensorDescriptor dsc_in;
    TensorDescriptor dsc_out;
};