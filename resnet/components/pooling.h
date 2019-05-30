#pragma once
#include "../helper/common.h"
#include "descriptor.h"

class PoolingFunctor {
  public:
    PoolingFunctor(dim_t in, int K, int padding, int stride)
        : dsc_pool(K, padding, stride), dsc_in(in) {
        init();
    }
    void init() {
        int n, c, h, w;
        auto st = cudnnGetPooling2dForwardOutputDim(dsc_pool, dsc_in, &n, &c, &h, &w);
        dsc_out.init(dim_t{n, c, h, w});
        check(st);
    }
    void forward(float* out, const float* in) {
        float alpha = 1.0;
        float beta = 0.0;
        auto st = cudnnPoolingForward(global.cudnn_handle(), dsc_pool, &alpha, dsc_in, in,
                                      &beta, dsc_out, out);
        check(st);
    }
    dim_t dims_out() {
        return dsc_out.dims();
    }
    void backward(float* in_grad, const float* in, const float* out_grad,
                  const float* out) {
        float alpha = 1.0;
        float beta = 1.0;
        auto st =
            cudnnPoolingBackward(global.cudnn_handle(), dsc_pool, &alpha, dsc_out, out,
                                 dsc_out, out_grad, dsc_in, in, &beta, dsc_in, in_grad);
        check(st);
    }

  private:
    PoolingDescriptor dsc_pool;
    TensorDescriptor dsc_in;
    TensorDescriptor dsc_out;
};