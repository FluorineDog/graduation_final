#pragma once
#include "../helper/common.h"
#include "../helper/wrapper.h"
#include "../helper/global.h"
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
    TensorDescriptor() {
        cudnnCreateTensorDescriptor(&desc_);
    }
    explicit TensorDescriptor(dim_t dims) : TensorDescriptor() {
        init(dims);
    }
    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;
    operator cudnnTensorDescriptor_t() {
        return desc_;
    }
    const dim_t& dims() {
        return dims_;
    }
    ~TensorDescriptor() {
        cudnnDestroyTensorDescriptor(desc_);
    }
    void recover(){
        size_t size;
        cudnnGetTensorSizeInBytes(desc_, &size);
        assert(size < (1ULL <<31));
        dims_ = dim_t{1, (int)size / 4};
    }

    void init(dim_t dims) {
        dims_ = dims;
        if(dims_.size() < 4) {
            dims_.resize(4, 1);
        }
        auto strides = get_strides(dims_);
        auto status = cudnnSetTensorNdDescriptor(desc_, kDataType, dims_.size(), dims_, strides);
        assert(status == CUDNN_STATUS_SUCCESS);
    }

  private:
    cudnnTensorDescriptor_t desc_;
    dim_t dims_;
};

class FilterDescriptor {
  public:
    FilterDescriptor() {
        cudnnCreateFilterDescriptor(&desc_);
    }
    explicit FilterDescriptor(dim_t dims) : FilterDescriptor() {
        init(dims);
    }

    FilterDescriptor(const FilterDescriptor&) = delete;
    FilterDescriptor& operator=(const FilterDescriptor&) = delete;
    operator cudnnFilterDescriptor_t() {
        return desc_;
    }
    const dim_t& dims() {
        return dims_;
    }
    ~FilterDescriptor() {
        cudnnDestroyFilterDescriptor(desc_);
    }

    void init(dim_t dims) {
        dims_ = dims;
        cudnnSetFilterNdDescriptor(desc_, kDataType, kFilterFormat, 4, dims_);
    }

  private:
    cudnnFilterDescriptor_t desc_;
    dim_t dims_;
};

class ConvolutionDescriptor {
  public:
    ConvolutionDescriptor(int padding, int stride, int dilation, int group)
        : padding_(padding), stride_(stride), dilation_(dilation), group_(group) {
        assert(dilation == 1);
        cudnnCreateConvolutionDescriptor(&desc_);
        init();
    }
    ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
    ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;
    operator cudnnConvolutionDescriptor_t() {
        return desc_;
    }
    ~ConvolutionDescriptor() {
        cudnnDestroyConvolutionDescriptor(desc_);
    }

  private:
    void init() {
        // auto dual_pack = [](int x) { return dim_t{x, x}; };
        cudnnSetConvolution2dDescriptor(desc_, padding_, padding_, stride_, stride_,
                                        dilation_, dilation_, CUDNN_CONVOLUTION,
                                        kDataType);
        // cudnnSetConvolutionGroupCount(desc_, group_);
    }

    cudnnConvolutionDescriptor_t desc_;
    const int padding_;
    const int stride_;
    const int dilation_;
    const int group_;
};

class ActivationDescriptor {
  public:
    explicit ActivationDescriptor() {
        cudnnCreateActivationDescriptor(&desc_);
        auto kMode = CUDNN_ACTIVATION_RELU;
        auto kNan = CUDNN_PROPAGATE_NAN;
        cudnnSetActivationDescriptor(desc_, kMode, kNan, 0.0);
    }
    ActivationDescriptor(const ActivationDescriptor&) = delete;
    ActivationDescriptor& operator=(const ActivationDescriptor&) = delete;
    ~ActivationDescriptor() {
        cudnnDestroyActivationDescriptor(desc_);
    }
    operator cudnnActivationDescriptor_t() {
        return desc_;
    }

  private:
    cudnnActivationDescriptor_t desc_;
};

class PoolingDescriptor {
  public:
    explicit PoolingDescriptor(int K, int padding, int stride) {
        cudnnCreatePoolingDescriptor(&desc_);
        auto kMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        auto kNan = CUDNN_PROPAGATE_NAN;
        cudnnSetPooling2dDescriptor(desc_, kMode, kNan, K, K, padding, padding, stride,
                                    stride);
    }
    PoolingDescriptor(const PoolingDescriptor&) = delete;
    PoolingDescriptor& operator=(const PoolingDescriptor&) = delete;
    ~PoolingDescriptor() {
        cudnnDestroyPoolingDescriptor(desc_);
    }
    operator cudnnPoolingDescriptor_t() {
        return desc_;
    }

  private:
    cudnnPoolingDescriptor_t desc_;
};