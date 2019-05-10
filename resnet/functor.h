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
    virtual size_t workspace_fwd() = 0;
    virtual size_t workspace_bwd_data() = 0;
    virtual size_t workspace_bwd_filter() = 0;
    virtual ~FunctorBase() {}
};

