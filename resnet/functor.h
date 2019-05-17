#pragma once
#include "descriptor.h"
#include "global.h"
// static T* FIX(float* ptr){
//     return static_cast<T*>(ptr);
// }
// static const T* FIX(const float* ptr){
//     return static_cast<const T*>(ptr);
// }

class FunctorBase {
  public:
    FunctorBase() = default;
    virtual void forward(const float* in, const float* weight, const float* in_grad) = 0;
    virtual void backward(const float* in, float* in_grad, const float* weight,
                          float* weight_grad, const float* out, float* out_grad) = 0;
    virtual size_t weight_size() = 0;
    virtual size_t workspace_size() = 0;
    virtual dim_t out_dims(dim_t input_dims) = 0;
    virtual ~FunctorBase() {}
};

class FunctorSpecial : public FunctorBase {
  public:
    FunctorSpecial(){}
    void forward(const float* in, const float* weight, const float* in_grad) {}
    void backward(const float* in, float* in_grad, const float* weight,
                  float* weight_grad, const float* out, float* out_grad) {}
    size_t weight_size() {return 0;}
    size_t workspace_size() {return 0;}
    dim_t out_dims(dim_t input_dims) {return input_dims;}
};