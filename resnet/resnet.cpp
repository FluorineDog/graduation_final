#include "common.h"
#include "wrapper.h"
#include "descriptor.h"
#include "global.h"
#include "functor.h"

using dim_t = Dims;
Global global;

void dog_print(std::string name, device_vector<T>& vec_vec, const dim_t& dim) {
    cout << name << endl;
    auto sz = get_volume(dim);
    host_vector<T> vec = vec_vec;
    auto tmp = dim;
    std::reverse(tmp.begin(), tmp.end());
    for(auto index : Range(sz)) {
        int index_cpy = index;
        for(auto x : tmp) {
            if(index_cpy % x != 0) break;
            index_cpy /= x;
            cout << "--------" << endl;
        }
        cout << vec[index] << " ";
    }
    cout << endl << "##########" << endl;
}

// template <class T>
void dog_resize_to(device_vector<T>& vec_vec, const dim_t& dim, bool set_value = false) {
    auto sz = get_volume(dim);
    vec_vec.resize(sz);
    if(set_value) {
        thrust::host_vector<T> host_vec(sz);
        for(auto id : Range(sz)) {
            host_vec[id] = (T)(id % 256);
        }
        vec_vec = host_vec;
    }
}

int main() {
    using T = float;
    DeviceVector<T> vec_in;
    DeviceVector<T> vec_filter;
    DeviceVector<T> vec_out;
    DeviceVector<T> vec_in_grad;
    DeviceVector<T> vec_filter_grad;
    DeviceVector<T> vec_out_grad;
    DeviceVector<char> vec_workspace;
    // thrust::host_vector<T> host_weight;
    // thrust::host_vector<T> host_in;
    // thrust::host_vector<T> host_out;
    constexpr int B = 1;
    constexpr int Ci = 2;
    constexpr int Co = 2;
    constexpr int W = 16;
    constexpr int H = 16;
    constexpr int K = 3;
    constexpr int group = 1;
    constexpr int padding = 1;
    constexpr int stride = 1;
    constexpr int dilation = 1;
    dim_t dims_in = {B, Ci, H, W};
    dim_t dims_filter = {Ci, Co, K, K};

    // dim_t dims_out = {B, Co, W, H};
    // dim_t dims_out =
    //     calc_dims_out(dims_in, dims_filter, group, padding, stride, dilation);
    // cudnnHandle_t handle;
    // TensorDescriptor dsc_in(dims_in);
    // TensorDescriptor dsc_out(dims_out);
    // FilterDescriptor dsc_filter(dims_filter);
    // ConvolutionDescriptor dsc_conv(padding, stride, dilation, group);
    ConvolutionFunctor functor(dims_in, dims_filter, group, padding, stride, dilation);
    auto dims_out = functor.get_dims_out();

    dog_resize_to(vec_in, dims_in, true);
    dog_resize_to(vec_in_grad, dims_in, false);
    dog_resize_to(vec_filter, dims_filter, true);
    dog_resize_to(vec_filter_grad, dims_filter, false);
    dog_resize_to(vec_out, dims_out, false);
    dog_resize_to(vec_out_grad, dims_out, true);
    // cudnnCreate(&handle);
    global.update_workspace_size(functor.workspace_fwd());
    global.update_workspace_size(functor.workspace_bwd_data());
    global.update_workspace_size(functor.workspace_bwd_filter());

    functor.forward(vec_out, vec_in, vec_filter);
    cudaDeviceSynchronize();

    functor.backwardData(vec_in_grad, vec_out_grad, vec_filter);
    cudaDeviceSynchronize();

    functor.backwardFilter(vec_filter_grad, vec_out_grad, vec_in);
    cudaDeviceSynchronize();

    // // conv pass
    // {
    //     auto kAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    //     size_t workspace_size;
    //     cudnnGetConvolutionForwardWorkspaceSize(handle, dsc_in, dsc_filter, dsc_conv,
    //                                             dsc_out, kAlgo, &workspace_size);
    //     vec_workspace.resize(workspace_size);
    //     float alpha = 1, beta = 0;
    //     cudnnConvolutionForward(handle, &alpha,                   //
    //                             dsc_in, vec_in,                   //
    //                             dsc_filter, vec_filter,           //
    //                             dsc_conv, kAlgo,                  //
    //                             vec_workspace, workspace_size,    //
    //                             &beta,                            //
    //                             dsc_out, vec_out                  //
    //     );
    //     cudaDeviceSynchronize();
    // }
    // // dgrad pass
    // {
    //     auto kAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    //     size_t workspace_size;
    //     cudnnGetConvolutionBackwardDataWorkspaceSize(
    //         handle, dsc_filter, dsc_out, dsc_conv, dsc_in, kAlgo, &workspace_size);
    //     if(workspace_size > vec_workspace.size()) {
    //         vec_workspace.resize(workspace_size * 1.5);
    //     }
    //     float alpha = 1, beta = 0;
    //     cudnnConvolutionBackwardData(handle, &alpha, dsc_filter, vec_filter, dsc_out,
    //                                  vec_out_grad, dsc_conv, kAlgo, vec_workspace,
    //                                  workspace_size, &beta, dsc_in, vec_in_grad);

    //     cudaDeviceSynchronize();
    // }
    // // Wgrad pass
    // {
    //     auto kAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    //     size_t workspace_size;
    //     cudnnGetConvolutionBackwardFilterWorkspaceSize(
    //         handle, dsc_in, dsc_out, dsc_conv, dsc_filter, kAlgo, &workspace_size);
    //     if(workspace_size > vec_workspace.size()) {
    //         vec_workspace.resize(workspace_size * 1.5);
    //     }
    //     float alpha = 1, beta = 0;
    //     cudnnConvolutionBackwardFilter(
    //         handle, &alpha, dsc_in, vec_in, dsc_out, vec_out_grad, dsc_conv, kAlgo,
    //         vec_workspace, workspace_size, &beta, dsc_filter, vec_filter_grad);
    //     cudaDeviceSynchronize();
    // }

    dog_print("input", vec_in, dims_in);
    dog_print("filter", vec_filter, dims_filter);
    dog_print("output", vec_out, dims_out);

    dog_print("input", vec_in_grad, dims_in);
    dog_print("filter", vec_filter_grad, dims_filter);
    dog_print("output", vec_out_grad, dims_out);
    return 0;
}
