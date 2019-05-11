#pragma once
#include "descriptor.h"
#include "global.h"
// static T* FIX(void* ptr){
//     return static_cast<T*>(ptr);
// }
// static const T* FIX(const void* ptr){
//     return static_cast<const T*>(ptr);
// }
// class FunctorBase {
//   public:
//     // FunctorBase() = default;
//     // virtual void forward(void* out, const void* in, const void* weight) = 0;
//     // virtual void backwardData(void* in_grad, const void* out_grad,
//     //                           const void* weight) = 0;
//     // virtual void backwardFilter(void* weight_grad, const void* out_grad,
//     //                             const void* in) = 0;
//     // virtual size_t weight_size() = 0;
//     // virtual size_t workspace_fwd() = 0;
//     // virtual size_t workspace_bwd_data() = 0;
//     // virtual size_t workspace_bwd_filter() = 0;
//     virtual ~FunctorBase() {}
// };

class BatchNorm  {
  public:
    static constexpr auto kMode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    BatchNorm(dim_t dims) : io_dsc(dims), fwd_counter(0) {
        cudnnDeriveBNTensorDescriptor(bn_dsc, io_dsc, kMode);
        this->bn_size = get_volume(bn_dsc.dims);
        extra.resize(bn_size * 4);
        thrust::fill_n(extra.begin(), extra.size(), 0);
    }
    void forward(void* out, const void* in, const void* weight)  {
        float alpha = 1, beta = 0;
        auto bnScale = (T*)weight;
        auto bnBias = 1 * bn_size + (T*)weight;
        auto bnRunningMean = (T*)extra;
        auto bnRunningVar = 1 * bn_size + (T*)extra;
        auto bnSavedMean = 2 * bn_size + (T*)extra;
        auto bnSavedVar = 3 * bn_size + (T*)extra;
        ++fwd_counter;
        cudnnBatchNormalizationForwardTraining(
            global.get_handle(), kMode, &alpha, &beta, io_dsc, in, io_dsc, out, bn_dsc,
            bnScale, bnBias, 1.0 / fwd_counter, bnRunningMean, bnRunningVar,
            CUDNN_BN_MIN_EPSILON, bnSavedMean, bnSavedVar);
    }
    size_t weight_size() override {
        return 2 * bn_size;
    }
    void backwardData(void* in_grad, const void* out_grad, const void* weight){
        float alpha = 1, beta = 0;

    }
    void backwardFilter(void* weight_grad, const void* out_grad, const void* in){};
    size_t workspace_fwd() {
        return 0;
    }
    size_t workspace_bwd_data() {
        return 0;
    }
    size_t workspace_bwd_filter() {
        return 0;
    }

  private:
    TensorDescriptor io_dsc;
    TensorDescriptor bn_dsc;
    DeviceVector<T> extra;
    size_t fwd_counter;
    size_t bn_size;
};
