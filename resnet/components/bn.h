#pragma once
#include "../descriptor.h"

class BatchNorm {
  public:
    static constexpr auto kMode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    BatchNorm(dim_t dims) : dsc_io(dims), fwd_counter(0) {
        cudnnDeriveBNTensorDescriptor(dsc_bn, dsc_io, kMode);
        this->bn_size = get_volume(dsc_bn.dims());
        extra.resize(bn_size * 4);
        thrust::fill_n(extra.begin(), extra.size(), 0);
    }
    void forward(float* out, const float* in, const float* weight) {
        float alpha = 1, beta = 0;
        auto bnScale = (const T*)weight;
        auto bnBias = 1 * bn_size + (const T*)weight;
        auto bnRunningMean = (T*)extra;
        auto bnRunningVar = 1 * bn_size + (T*)extra;
        auto bnSavedMean = 2 * bn_size + (T*)extra;
        auto bnSavedVar = 3 * bn_size + (T*)extra;
        ++fwd_counter;
        cudnnBatchNormalizationForwardTraining(
            global.cudnn_handle(), kMode, &alpha, &beta, dsc_io, in, dsc_io, out, dsc_bn,
            bnScale, bnBias, 1.0 / fwd_counter, bnRunningMean, bnRunningVar,
            CUDNN_BN_MIN_EPSILON, bnSavedMean, bnSavedVar);
    }
    size_t weight_size() {
        return 2 * bn_size;
    }
    dim_t out_dim() {
        return dsc_io.dims();
    }
    void backward(float* in_grad, float* weight_grad, const float* in,
                  const float* out_grad, const float* weight) {
        float alpha = 1, betaOut = 1;
        // float betaIn = 0;
        auto bnScale = (const T*)weight;
        // auto bnBias = 1 * bn_size + (T*)weight;
        auto bnScale_grad = (T*)weight_grad;
        auto bnBias_grad = 1 * bn_size + (T*)weight_grad;
        auto bnSavedMean = 2 * bn_size + (T*)extra;
        auto bnSavedVar = 3 * bn_size + (T*)extra;
        cudnnBatchNormalizationBackward(
            global.cudnn_handle(), kMode, &alpha, &betaOut, &alpha, &betaOut, dsc_io, in,
            dsc_io, out_grad, dsc_io, in_grad, dsc_bn, bnScale, bnScale_grad, bnBias_grad,
            CUDNN_BN_MIN_EPSILON, bnSavedMean, bnSavedVar);
    }

  private:
    TensorDescriptor dsc_io;
    TensorDescriptor dsc_bn;
    DeviceVector<T> extra;
    size_t fwd_counter;
    size_t bn_size;
};
