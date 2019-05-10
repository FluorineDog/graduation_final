#pragma once
#include "descriptor.h"
#include "global.h"
// static T* FIX(void* ptr){
//     return static_cast<T*>(ptr);
// }
// static const T* FIX(const void* ptr){
//     return static_cast<const T*>(ptr);
// }

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

class ConvolutionFunctor {
  public:
    ConvolutionFunctor(dim_t dims_in, dim_t dims_filter, int group, int padding,
                       int stride, int dilation)
        : dsc_conv(padding, stride, dilation, group),
          dsc_in(dims_in),
          dsc_filter(dims_filter) {
        dim_t dims_out =
            calc_dims_out(dims_in, dims_filter, group, padding, stride, dilation);
        dsc_out.init(dims_out);
    }
    void forward(void* out, const void* in, const void* filter) {
        float alpha = 1, beta = 0;
        auto kAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        cudnnConvolutionForward(global.get_handle(), &alpha,                            //
                                dsc_in, in,                                             //
                                dsc_filter, filter,                                     //
                                dsc_conv, kAlgo,                                        //
                                global.get_workspace(), global.get_workspace_size(),    //
                                &beta,                                                  //
                                dsc_out, out                                            //
        );
    }
    size_t workspace_fwd() {
        auto kAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        size_t workspace_size;
        cudnnGetConvolutionForwardWorkspaceSize(global.get_handle(), dsc_in, dsc_filter,
                                                dsc_conv, dsc_out, kAlgo,
                                                &workspace_size);
        return workspace_size;
    }
    void backwardData(void* in_grad, const void* out_grad, const void* filter) {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        float alpha = 1, beta = 0;
        cudnnConvolutionBackwardData(global.get_handle(), &alpha, dsc_filter, filter,
                                     dsc_out, out_grad, dsc_conv, kAlgo,
                                     global.get_workspace(), global.get_workspace_size(),
                                     &beta, dsc_in, in_grad);
    }
    size_t workspace_bwd_data() {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        size_t workspace_size;
        cudnnGetConvolutionBackwardDataWorkspaceSize(global.get_handle(), dsc_filter,
                                                     dsc_out, dsc_conv, dsc_in, kAlgo,
                                                     &workspace_size);
        return workspace_size;
    }
    void backwardFilter(void* filter_grad, const void* out_grad, const void* in) {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        float alpha = 1, beta = 0;
        cudnnConvolutionBackwardFilter(global.get_handle(), &alpha, dsc_in, in, dsc_out,
                                       out_grad, dsc_conv, kAlgo, global.get_workspace(),
                                       global.get_workspace_size(), &beta, dsc_filter,
                                       filter_grad);
    }
    size_t workspace_bwd_filter() {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        size_t workspace_size;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(global.get_handle(), dsc_in,
                                                       dsc_out, dsc_conv, dsc_filter,
                                                       kAlgo, &workspace_size);
        return workspace_size;
    }
    dim_t get_dims_out() {
        return dsc_out.dims();
    }

  private:
    ConvolutionDescriptor dsc_conv;
    TensorDescriptor dsc_in;
    TensorDescriptor dsc_out;
    FilterDescriptor dsc_filter;
};
