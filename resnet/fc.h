#pragma once
#include "functor.h"
#include "cublas.h"
#include "cu_extra.h"

inline void sgemm(bool transA, bool transB, int m, int k, int n, const float* a,
                  const float* b, float* c, bool acc = false) {
    auto kTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto kTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto lda = transA ? m : k;
    auto ldb = transB ? k : n;
    auto ldc = n;

    float float_one = 1.0;
    float float_acc = acc ? 1.0 : 0.0;
    // cout << transA << endl;
    // cout << k << endl;
    // cout << lda << endl;
    cublasSgemm_v2(global.cublas_handle(), kTransB, kTransA, n, m, k, &float_one, b, ldb,
                   a, lda, &float_acc, c, ldc);
}

class FCFunctor {
  public:
    FCFunctor(int batch, int in_size, int out_size)
        : batch(batch), in_size(in_size), out_size(out_size) {
        ones.resize(batch);
        thrust::fill_n(ones.begin(), batch, 1.0);
    }
    void forward(float* out, const float* in, const float* weight) {
        auto bias = weight + in_size * out_size;
        // add_biased(out, out_size, batch, 1.0, bias);
        // float float_one = 1.0;
        // float float_zero = 0.0;
        // cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, batch, 1,
        //                out_size, &float_one, bias, 1, ones, out_size, &float_zero, out,
        //                out_size);
        // batch*out <= batch*1 x 1*out
        sgemm(false, false, batch, 1, out_size, bias, ones, out);
        // batch*out <= batch*in x in*out
        // cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, batch, in_size,
        //                out_size, &float_one, in, in_size, weight, out_size, &float_one,
        //                out, out_size);
        sgemm(false, false, batch, in_size, out_size, in, weight, out, false);
    }

    void backward(float* in_grad, float* weight_grad, const float* in,
                  const float* out_grad, const float* weight) {
        backwardFilter(weight_grad, in, out_grad);
        if(in_grad) {
            backwardData(in_grad, weight, out_grad);
        }
    }

    size_t size_parameters() {
        return in_size * out_size + out_size;
    }
    dim_t out_dim() {
        return dim_t{batch, out_size};
    }

  private:
    void backwardFilter(float* weight_grad, const float* in, const float* out_grad) {
        auto bias_grad = weight_grad + in_size * out_size;
        // float float_one = 1.0;
        float float_zero = 0.0;
        // W: inxout <= batch*in x batch*out
        sgemm(true, false, in_size, batch, out_size, in, out_grad, weight_grad);
        // cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, in_size, batch,
        //                out_size, &float_one, in, in_size, out_grad, out_size, &float_zero,
        //                out_grad, out_size);
        // b: out <= batch*out * batch
        cublasSgemv_v2(global.cublas_handle(), CUBLAS_OP_N, out_size, batch, &float_zero,
                       out_grad, out_size, ones, 1, &float_zero, bias_grad, 1);
    }
    void backwardData(float* in_grad_, const float* weight_, const float* out_grad_) {
        // in: batchxin: batchxout * in*out
        auto in_grad = static_cast<float*>(in_grad_);
        auto out_grad = static_cast<const float*>(out_grad_);
        auto weight = static_cast<const float*>(weight_);
        // auto bias = weight + in_size * out_size;
        // float float_one = 1.0;
        // float float_zero = 0.0;
        sgemm(false, true, batch, out_size, in_size, out_grad, weight, in_grad);
        // cublasSgemm_v2(global.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, batch, out_size,
        //                in_size, &float_one, out_grad, out_size, weight, out_size,
        //                &float_zero, in_grad, in_size);
    }

  private:
    DeviceVector<float> ones;
    int batch;
    int in_size;
    int out_size;
};