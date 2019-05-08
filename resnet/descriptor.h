#pragma once
#include "common.h"
#include "wrapper.h"
using dim_t = Dims;

inline ull get_volume(const dim_t& vec) {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<ull>());
}

inline dim_t get_strides(const dim_t& vec) {
    dim_t tmp(vec.size());
    int acc = 1;
    for(auto iter : Range(vec.size())) {
        auto cur = vec.size() - 1 - iter;
        tmp[cur] = acc;
        acc *= vec[cur];
    }
    return tmp;
}

constexpr auto kDataType = CUDNN_DATA_FLOAT;
constexpr auto kFilterFormat = CUDNN_TENSOR_NCHW;
class TensorDescriptor {
  public:
    TensorDescriptor(dim_t dims) : dims_(dims) {
        cudnnCreateTensorDescriptor(&desc_);
        init();
    }
    operator cudnnTensorDescriptor_t() {
        return desc_;
    }
    const dim_t& dims() {
        return dims_;
    }
    ~TensorDescriptor() {
        cudnnDestroyTensorDescriptor(desc_);
    }

  private:
    void init() {
        auto strides = get_strides(dims_);
        cudnnSetTensorNdDescriptor(desc_, kDataType, 4, dims_, strides);
    }
    cudnnTensorDescriptor_t desc_;
    dim_t dims_;
};

class FilterDescriptor {
  public:
    FilterDescriptor(dim_t dims) : dims_(dims) {
        cudnnCreateFilterDescriptor(&desc_);
        init();
    }
    operator cudnnFilterDescriptor_t() {
        return desc_;
    }
    const dim_t& dims() {
        return dims_;
    }
    ~FilterDescriptor() {
        cudnnDestroyFilterDescriptor(desc_);
    }

  private:
    void init() {
        cudnnSetFilterNdDescriptor(desc_, kDataType, kFilterFormat, 4, dims_);
    }
    cudnnFilterDescriptor_t desc_;
    dim_t dims_;
};
