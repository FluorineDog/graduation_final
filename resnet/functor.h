#pragma once
#include "descriptor.h"
#include "global.h"
// static T* FIX(void* ptr){
//     return static_cast<T*>(ptr);
// }
// static const T* FIX(const void* ptr){
//     return static_cast<const T*>(ptr);
// }
class FunctorBase {
  public:
    FunctorBase() = default;
    virtual void forward(void* out, const void* in, const void* weight) = 0;
    virtual void backwardData(void* in_grad, const void* out_grad,
                              const void* weight) = 0;
    virtual void backwardFilter(void* weight_grad, const void* out_grad,
                                const void* in) = 0;
    virtual size_t weight_size() = 0;
    virtual size_t workspace_fwd() = 0;
    virtual size_t workspace_bwd_data() = 0;
    virtual size_t workspace_bwd_filter() = 0;
    virtual ~FunctorBase() {}
};

class BatchNorm : public FunctorBase {
  public:
    BatchNorm(dim_t dims) : io_dsc(dims), bn_dsc({1, dims[1], 1, 1}) {}
    void forward(void* out, const void* in, const void* weight) override {
        auto kMode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
        float alpha = 1, beta = 0;
        auto bnScale = (T*)weight;
        auto bnBias = bn_dsc.dims[1] + (T*)weight;
        cudnnBatchNormalizationForwardTraining(global.get_handle(), kMode, &alpha, &beta,
                                               io_dsc, in, io_dsc, out, bn_dsc, bnScale, bnBias, );
    }
    size_t weight_size() override {
        dim_t dims = dsc.dims();
        return 2 * dims[1] * sizeof(T);
    }
    void backwardData(void* in_grad, const void* out_grad, const void* weight) = 0;
    void backwardFilter(void* weight_grad, const void* out_grad, const void* in) = 0;
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
};
