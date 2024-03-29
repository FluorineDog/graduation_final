#pragma once
#include "functor.h"

// assume dilation = 1

class ConvolutionFunctor {
  public:
    ConvolutionFunctor(
        dim_t dims_in, int C_out, int K, int group, int padding, int stride, int dilation)
        : dsc_conv(padding, stride, dilation, group),
          dsc_in(dims_in),
          dsc_filter(dim_t{C_out, dims_in[1], K, K}),
          params_{group, padding, stride, dilation} {
        dim_t dims_out = calc_dims_out();
        dsc_out.init(dims_out);
        // if(stride != 1 || K != 3) {
        //     algo_.fwd = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
        //     algo_.bwd_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        //     algo_.bwd_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        // }
        set_algo();
    }

    void set_algo() {
        cudnnGetConvolutionBackwardDataAlgorithm(
            global.cudnn_handle(), dsc_filter, dsc_out, dsc_conv, dsc_in,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, (size_t)-1, &algo_.bwd_data);
        cudnnGetConvolutionBackwardFilterAlgorithm(
            global.cudnn_handle(), dsc_in, dsc_out, dsc_conv, dsc_filter,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, (size_t)-1, &algo_.bwd_filter);
        cudnnGetConvolutionForwardAlgorithm(
            global.cudnn_handle(), dsc_in, dsc_filter, dsc_conv, dsc_out,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, (size_t)-1, &algo_.fwd);
    }
    void forward(float* out, const float* in, const float* filter) {
        float alpha = 1, beta = 0;
        auto kAlgo = algo_.fwd;
        auto st = cudnnConvolutionForward(
            global.cudnn_handle(), &alpha,    //
            dsc_in, in,                       //
            dsc_filter, filter,               //
            dsc_conv, kAlgo,                  //
            global.get_workspace(),
            global.get_workspace_size(),    //
            &beta,                          //
            dsc_out, out                    //
        );
        check(st);
    }

    void backward(
        float* in_grad, float* weight_grad, const float* in, const float* out_grad,
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
        auto k0 = workspace_fwd();
        auto k1 = workspace_bwd_data();
        auto k2 = workspace_bwd_filter();

        return std::max(std::max(k0, k1), k2);
    }

  private:
    size_t workspace_fwd() {
        auto kAlgo = algo_.fwd;
        size_t workspace_size = 0;
        auto st = cudnnGetConvolutionForwardWorkspaceSize(
            global.cudnn_handle(), dsc_in, dsc_filter, dsc_conv, dsc_out, kAlgo,
            &workspace_size);
        check(st);
        return workspace_size;
    }
    void backwardData(float* in_grad, const float* out_grad, const float* filter) {
        auto kAlgo = algo_.bwd_data;
        float alpha = 1, beta = 1.0;
        auto st = cudnnConvolutionBackwardData(
            global.cudnn_handle(), &alpha, dsc_filter, filter, dsc_out, out_grad,
            dsc_conv, kAlgo, global.get_workspace(), global.get_workspace_size(), &beta,
            dsc_in, in_grad);
        check(st);
    }

    size_t workspace_bwd_data() {
        auto kAlgo = algo_.bwd_data;
        size_t workspace_size;
        auto st = cudnnGetConvolutionBackwardDataWorkspaceSize(
            global.cudnn_handle(), dsc_filter, dsc_out, dsc_conv, dsc_in, kAlgo,
            &workspace_size);
        check(st);
        return workspace_size;
    }
    void backwardFilter(float* filter_grad, const float* out_grad, const float* in) {
        auto kAlgo = algo_.bwd_filter;
        float alpha = 1, beta = 1.0;
        auto st = cudnnConvolutionBackwardFilter(
            global.cudnn_handle(), &alpha, dsc_in, in, dsc_out, out_grad, dsc_conv, kAlgo,
            global.get_workspace(), global.get_workspace_size(), &beta, dsc_filter,
            filter_grad);
        check(st);
    }
    size_t workspace_bwd_filter() {
        auto kAlgo = algo_.bwd_filter;
        size_t workspace_size;
        auto st = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            global.cudnn_handle(), dsc_in, dsc_out, dsc_conv, dsc_filter, kAlgo,
            &workspace_size);
        // assert(status == CUDNN_STATUS_SUCCESS);
        check(st);
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
    dim_t calc_dims_out() {
        int n, c, h, w;
        auto status = cudnnGetConvolution2dForwardOutputDim(
            dsc_conv, dsc_in, dsc_filter, &n, &c, &h, &w);
        //     return (len - kernel + 2 * padding) / stride + 1;
        check(status);
        return dim_t{n, c, h, w};
    }

  private:
    Params params_;
    ConvolutionDescriptor dsc_conv;
    TensorDescriptor dsc_in;
    TensorDescriptor dsc_out;
    FilterDescriptor dsc_filter;
    struct {
        cudnnConvolutionFwdAlgo_t fwd = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        cudnnConvolutionBwdDataAlgo_t bwd_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
        cudnnConvolutionBwdFilterAlgo_t bwd_filter =
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
    } algo_;
};
