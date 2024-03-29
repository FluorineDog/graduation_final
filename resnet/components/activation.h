#pragma once
#include "functor.h"



class ActivationFunctor {
  public:
    ActivationFunctor(dim_t dims) : dsc_io(dims) {}
    void forward(float* out, const float* in) {
        float alpha = 1.0;
        float beta = 0.0;
        // dog_print("in", in, {10, 8});
        cudnnActivationForward(global.cudnn_handle(), dsc_act, &alpha, dsc_io, in, &beta,
                               dsc_io, out);
        // dog_print("in", out, {10, 8});
    }
    void backward(float* in_grad, const float* out_grad, const float* in,
                  const float* out) {
        float alpha = 1.0;
        float beta = 1.0;
        // need to be adjust
        cudnnActivationBackward(global.cudnn_handle(), dsc_act, &alpha, dsc_io, out,
                                dsc_io, out_grad, dsc_io, in, &beta, dsc_io, in_grad);
    }
    dim_t out_dim(){
        return dsc_io.dims();
    }
  private:
    ActivationDescriptor dsc_act;
    TensorDescriptor dsc_io;
};