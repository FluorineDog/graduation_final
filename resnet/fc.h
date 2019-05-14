#include "functor.h"
#include "cublas.h"
#include "cu_extra.h"

class FCFunctor {
  public:
    FCFunctor(int batch, int in_size, int out_size)
        : batch(batch), in_size(in_size), out_size(out_size) {
        ones.resize(batch);
        thrust::fill_n(ones.size(), batch, 1.0);
    }
    void forward(void* out_, const void* in_, const void* weight_) {
        auto out = (float*)out_;
        auto in = (float*)in_;
        auto weight = (float*)weight_;
        auto bias = weight + in_size * out_size;
        // add_biased(out, out_size, batch, 1.0, bias);
        float float_one = 1.0;
        float float_zero = 0.0;
        // batch*out <= batch*1 x 1*out
        cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, batch, 1,
                       out_size, &float_one, bias, 1, ones, out_size, &float_zero, out,
                       out_size);
        // batch*out <= batch*in x in*out
        cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, batch, in_size,
                       out_size, &float_one, in, in_size, weight, out_size, &float_one,
                       out, out_size);
    }
    void backwardFilter(void* weight_grad_, const void* in_, const void* out_grad_) {
        auto in = (float*)in_;
        auto out_grad = (float*)out_grad_;
        auto weight_grad = (float*)weight_grad_;
        auto bias_grad = weight_grad + in_size * out_size;
        float float_one = 1.0;
        float float_zero = 0.0;
        // W: inxout <= batch*in x batch*out
        cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, in_size, batch,
                       out_size, &float_one, in, in_size, out_grad, out_size, &float_zero,
                       out_grad, out_size);
        // b: out <= batch*out * batch
        cublasSgemv_v2(global.cublas_handle(), CUBLAS_OP_T, out_size, batch, &float_zero,
                       out_grad, out_size, ones, 1, &float_zero, bias_grad, 1);
    }
    void backwardData(void* in_grad_, const void* weight_, const void* out_grad_) {
        // in: batchxin: batchxout * in*out
        auto in_grad = (float*)in_grad_;
        auto out_grad = (float*)out_grad_;
        auto weight = (float*)weight_;
        auto bias = weight + in_size * out_size;
        float float_one = 1.0;
        float float_zero = 0.0;
        cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, batch, out_size,
                       in_size, &float_one, out_grad, out_size, weight, out_size,
                       &float_zero, in_grad, in_size);
    }
    void backward(void* in_grad, void* weight_grad, const void* in, const void* out_grad,
                  const void* weight) {
        backwardFilter(weight_grad, in, out_grad);
        backwardData(in_grad, weight, out_grad);
    }

  private:
    DeviceVector<float> ones;
    int batch;
    int in_size;
    int out_size;
};