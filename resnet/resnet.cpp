#include "common.h"
#include "wrapper.h"
#include "descriptor.h"
using dim_t = Dims;


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
    dim_t dims_out =
        calc_dims_out(dims_in, dims_filter, group, padding, stride, dilation);
    cudnnHandle_t handle;
    TensorDescriptor dsc_in(dims_in);
    TensorDescriptor dsc_out(dims_out);
    FilterDescriptor dsc_filter(dims_filter);
    
    cudnnConvolutionDescriptor_t dsc_conv;

    dog_resize_to(vec_in, dims_in, true);
    dog_resize_to(vec_in_grad, dims_in, false);
    dog_resize_to(vec_filter, dims_filter, true);
    dog_resize_to(vec_filter_grad, dims_filter, false);
    dog_resize_to(vec_out, dims_out, false);
    dog_resize_to(vec_out_grad, dims_out, true);
    cudnnCreate(&handle);
    // cudnnCreateTensorDescriptor(&dsc_in);
    // cudnnCreateTensorDescriptor(&dsc_out);
    // cudnnCreateFilterDescriptor(&dsc_filter);

    cudnnCreateConvolutionDescriptor(&dsc_conv);

    // auto strides_in = get_strides(dims_in);
    // auto strides_out = get_strides(dims_out);
    // auto strides_filter = get_strides(dims_filter);

    // cudnnSetTensorNdDescriptor(dsc_in, kDataType, 4, dims_in, strides_in);
    // cudnnSetTensorNdDescriptor(dsc_out, kDataType, 4, dims_out, strides_out);
    // cudnnSetFilterNdDescriptor(dsc_filter, kDataType, kFilterFormat, 4, dims_filter);

    auto dual_pack = [](int x) { return dim_t{x, x}; };
    cudnnSetConvolutionNdDescriptor(dsc_conv, 2, dual_pack(padding), dual_pack(stride),
                                    dual_pack(dilation), CUDNN_CONVOLUTION, kDataType);
    
    // conv pass
    {
        auto kAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        size_t workspace_size;
        cudnnGetConvolutionForwardWorkspaceSize(handle, dsc_in, dsc_filter, dsc_conv,
                                                dsc_out, kAlgo, &workspace_size);
        vec_workspace.resize(workspace_size);
        float alpha = 1, beta = 0;
        cudnnConvolutionForward(handle, &alpha,                   //
                                dsc_in, vec_in,                   //
                                dsc_filter, vec_filter,           //
                                dsc_conv, kAlgo,                  //
                                vec_workspace, workspace_size,    //
                                &beta,                            //
                                dsc_out, vec_out                  //
        );
        cudaDeviceSynchronize();
    }
    // dgrad pass
    {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        size_t workspace_size;
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, dsc_filter, dsc_out, dsc_conv, dsc_in, kAlgo, &workspace_size);
        if(workspace_size > vec_workspace.size()) {
            vec_workspace.resize(workspace_size * 1.5);
        }
        float alpha = 1, beta = 0;
        cudnnConvolutionBackwardData(handle, &alpha, dsc_filter, vec_filter, dsc_out,
                                     vec_out_grad, dsc_conv, kAlgo, vec_workspace,
                                     workspace_size, &beta, dsc_in, vec_in_grad);

        cudaDeviceSynchronize();
    }
    // Wgrad pass
    {
        auto kAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        size_t workspace_size;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, dsc_in, dsc_out, dsc_conv, dsc_filter, kAlgo, &workspace_size);
        if(workspace_size > vec_workspace.size()) {
            vec_workspace.resize(workspace_size * 1.5);
        }
        float alpha = 1, beta = 0;
        cudnnConvolutionBackwardFilter(
            handle, &alpha, dsc_in, vec_in, dsc_out, vec_out_grad, dsc_conv, kAlgo,
            vec_workspace, workspace_size, &beta, dsc_filter, vec_filter_grad);
        cudaDeviceSynchronize();
    }

    dog_print("input", vec_in, dims_in);
    dog_print("filter", vec_filter, dims_filter);
    dog_print("output", vec_out, dims_out);

    dog_print("input", vec_in_grad, dims_in);
    dog_print("filter", vec_filter_grad, dims_filter);
    dog_print("output", vec_out_grad, dims_out);
    return 0;
}
