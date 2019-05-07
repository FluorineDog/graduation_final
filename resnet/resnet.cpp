
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "doglib/common/common.h"
#include <cudnn.h>
#include <cuda_runtime.h>
using namespace doglib::common;
using ull = long long;
using T = float;
using dim_t = std::vector<int>;
using namespace thrust;

ull get_volume(const dim_t& vec) {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<ull>());
}

dim_t get_strides(const dim_t& vec) {
    dim_t tmp(vec.size());
    int acc = 1;
    for(auto iter : Range(vec.size())) {
        auto cur = vec.size() - 1 - iter;
        tmp[cur] = acc;
        acc *= vec[cur];
    }
    return tmp;
}

// template <class T>
void dog_resize_to(device_vector<T>& dev_vec, const dim_t& dim, bool set_value = false) {
    auto sz = get_volume(dim);
    dev_vec.resize(sz);
    if(set_value) {
        thrust::host_vector<T> host_vec(sz);
        for(auto id : Range(sz)) {
            host_vec[id] = (T)id;
        }
        dev_vec = host_vec;
    }
}

// assume dilation = 1
dim_t calc_dims_out(             //
    const dim_t& dims_in,        //
    const dim_t& dims_filter,    //
    int group,                   //
    int padding,                 //
    int stride,                  //
    int dilation                 //
) {
    assert(dims_in.size() == 4);
    assert(dims_filter.size() == 4);
    assert(dims_in[1] == dims_filter[0]);
    assert(group == 1);       // todo
    assert(dilation == 1);    // todo
    dim_t output(4);
    // B
    output[0] = dims_in[0];
    // Co
    output[1] = dims_filter[1];
    auto gen_len = [=](int len, int kernel) {
        return (len - kernel + 2 * padding) / stride + 1;
    };
    // H, W
    output[2] = gen_len(dims_in[2], dims_filter[2]);
    output[3] = gen_len(dims_in[3], dims_filter[3]);
    return output;
}

int main() {
    using T = float;
    thrust::device_vector<T> dev_in;
    thrust::device_vector<T> dev_filter;
    thrust::device_vector<T> dev_out;
    thrust::device_vector<char> dev_workspace;
    // thrust::host_vector<T> host_weight;
    // thrust::host_vector<T> host_in;
    // thrust::host_vector<T> host_out;
    constexpr int B = 1;
    constexpr int Ci = 32;
    constexpr int Co = 32;
    constexpr int W = 32;
    constexpr int H = 32;
    constexpr int K = 3;
    constexpr int group = 1;
    constexpr int padding = 1;
    constexpr int stride = 1;
    constexpr int dilation = 1;
    constexpr auto kDataType = CUDNN_DATA_FLOAT;
    constexpr auto kFilterFormat = CUDNN_TENSOR_NCHW;
    dim_t dims_in = {B, Ci, H, W};
    dim_t dims_filter = {Ci, Co, K, K};
    // dim_t dims_out = {B, Co, W, H};
    dim_t dims_out =
        calc_dims_out(dims_in, dims_filter, group, padding, stride, dilation);
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dsc_in;
    cudnnFilterDescriptor_t dsc_filter;
    cudnnTensorDescriptor_t dsc_out;
    cudnnConvolutionDescriptor_t dsc_conv;

    dog_resize_to(dev_in, dims_in, true);
    dog_resize_to(dev_filter, dims_filter, true);
    dog_resize_to(dev_out, dims_out, false);
    cudnnCreate(&handle);
    cudnnCreateTensorDescriptor(&dsc_in);
    cudnnCreateFilterDescriptor(&dsc_filter);
    cudnnCreateTensorDescriptor(&dsc_out);
    cudnnCreateConvolutionDescriptor(&dsc_conv);

    auto strides_in = get_strides(dims_in);
    auto strides_filter = get_strides(dims_filter);
    auto strides_out = get_strides(dims_out);

    cudnnSetTensorNdDescriptor(dsc_in, kDataType, 4, dims_in.data(), strides_in.data());
    cudnnSetTensorNdDescriptor(dsc_out, kDataType, 4, dims_out.data(),
                               strides_out.data());
    auto dual_pack = [](int x) { return dim_t{x, x}; };
    cudnnSetConvolutionNdDescriptor(dsc_conv, 2, dual_pack(padding).data(),
                                    dual_pack(stride).data(), dual_pack(dilation).data(),
                                    CUDNN_CONVOLUTION, kDataType);
    //
    cudnnSetFilterNdDescriptor(dsc_filter, kDataType, kFilterFormat, 4,
                               dims_filter.data());
    auto kAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    {
        size_t workspace_size;
        cudnnGetConvolutionForwardWorkspaceSize(handle, dsc_in, dsc_filter, dsc_conv,
                                                dsc_out, kAlgo, &workspace_size);
        dev_workspace.resize(workspace_size);
    }
    float alpha = 1, beta = 0;
    cudnnConvolutionForward(handle, &alpha,                                      //
                            dsc_in, dev_in.data().get(),                         //
                            dsc_filter, dev_filter.data().get(),                 //
                            dsc_conv, kAlgo,                                     //
                            dev_workspace.data().get(), dev_workspace.size(),    //
                            &beta,                                               //
                            dsc_out, dev_out.data().get()                        //
    );
    cudaDeviceSynchronize();
    
    return 0;
}