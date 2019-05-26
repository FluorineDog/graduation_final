#pragma once
#include "functor.h"

// assume dilation = 1
static dim_t calc_dims_out(      //
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
    ConvolutionFunctor(dim_t dims_in, int C_out, int K, int group, int padding,
                       int stride, int dilation)
        : dsc_conv(padding, stride, dilation, group),
          dsc_in(dims_in),
          dsc_filter(dim_t{dims_in[1], C_out, K, K}),
          params_{group, padding, stride, dilation} {
        dim_t dims_out =
            calc_dims_out(dsc_in.dims(), dsc_filter.dims(), group, padding, stride, dilation);
        dsc_out.init(dims_out);
    }
    void forward(float* out, const float* in, const float* filter) {
        float alpha = 1, beta = 0;
        auto kAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        cudnnConvolutionForward(global.cudnn_handle(), &alpha,                          //
                                dsc_in, in,                                             //
                                dsc_filter, filter,                                     //
                                dsc_conv, kAlgo,                                        //
                                global.get_workspace(), global.get_workspace_size(),    //
                                &beta,                                                  //
                                dsc_out, out                                            //
        );
    }

    void backward(float* in_grad, float* weight_grad, const float* in, const float* out_grad,
                  const float* weight) {
        if(in_grad) {
            backwardData(in_grad, out_grad, weight);
        }
        backwardFilter(weight_grad, out_grad, in);
    }

    dim_t dims_out() {
        // TODO: Change to cudnn routine
        return dsc_out.dims();
    }

    size_t get_weight_size() {
        return get_volume(dsc_filter.dims());
    }

    size_t get_workspace_size() {
        return std::max(workspace_bwd_data(),
                        std::max(workspace_bwd_filter(), workspace_fwd()));
    }

  private:
    size_t workspace_fwd() {
        auto kAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        size_t workspace_size;
        cudnnGetConvolutionForwardWorkspaceSize(global.cudnn_handle(), dsc_in, dsc_filter,
                                                dsc_conv, dsc_out, kAlgo,
                                                &workspace_size);
        return workspace_size;
    }
    void backwardData(float* in_grad, const float* out_grad, const float* filter) {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        float alpha = 1, beta = 1.0;
        cudnnConvolutionBackwardData(global.cudnn_handle(), &alpha, dsc_filter, filter,
                                     dsc_out, out_grad, dsc_conv, kAlgo,
                                     global.get_workspace(), global.get_workspace_size(),
                                     &beta, dsc_in, in_grad);
    }

    size_t workspace_bwd_data() {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        size_t workspace_size;
        cudnnGetConvolutionBackwardDataWorkspaceSize(global.cudnn_handle(), dsc_filter,
                                                     dsc_out, dsc_conv, dsc_in, kAlgo,
                                                     &workspace_size);
        return workspace_size;
    }
    void backwardFilter(float* filter_grad, const float* out_grad, const float* in) {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        float alpha = 1, beta = 1.0;
        cudnnConvolutionBackwardFilter(global.cudnn_handle(), &alpha, dsc_in, in, dsc_out,
                                       out_grad, dsc_conv, kAlgo, global.get_workspace(),
                                       global.get_workspace_size(), &beta, dsc_filter,
                                       filter_grad);
    }
    size_t workspace_bwd_filter() {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        size_t workspace_size;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(global.cudnn_handle(), dsc_in,
                                                       dsc_out, dsc_conv, dsc_filter,
                                                       kAlgo, &workspace_size);
        return workspace_size;
    }

    struct Params {
        int group;
        int padding;
        int stride;
        int dilation;
    };

    const Params& get_params() {
        return params_;
    }

  private:
    Params params_;
    ConvolutionDescriptor dsc_conv;
    TensorDescriptor dsc_in;
    TensorDescriptor dsc_out;
    FilterDescriptor dsc_filter;
};
