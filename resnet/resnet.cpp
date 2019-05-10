#include "common.h"
#include "wrapper.h"
#include "descriptor.h"
#include "global.h"
#include "conv.h"
#include "../doglib/time/timer.h"

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

    constexpr int B = 4;
    constexpr int Ci = 512;
    constexpr int Co = 512;
    constexpr int W = 64;
    constexpr int H = 64;
    constexpr int K = 3;
    constexpr int group = 1;
    constexpr int padding = 1;
    constexpr int stride = 1;
    constexpr int dilation = 1;
    dim_t dims_in = {B, Ci, H, W};
    dim_t dims_filter = {Ci, Co, K, K};
    ConvolutionFunctor functor(dims_in, dims_filter, group, padding, stride, dilation);
    auto dims_out = functor.get_dims_out();

    dog_resize_to(vec_in, dims_in, true);
    dog_resize_to(vec_in_grad, dims_in, false);
    dog_resize_to(vec_filter, dims_filter, true);
    dog_resize_to(vec_filter_grad, dims_filter, false);
    dog_resize_to(vec_out, dims_out, false);
    dog_resize_to(vec_out_grad, dims_out, true);

    global.update_workspace_size(functor.workspace_fwd());
    global.update_workspace_size(functor.workspace_bwd_data());
    global.update_workspace_size(functor.workspace_bwd_filter());
    doglib::time::Timer timer;
    functor.forward(vec_out, vec_in, vec_filter);
    cudaDeviceSynchronize();
    
    functor.backwardData(vec_in_grad, vec_out_grad, vec_filter);
    cudaDeviceSynchronize();

    functor.backwardFilter(vec_filter_grad, vec_out_grad, vec_in);
    cudaDeviceSynchronize();

    dog_print("input", vec_in, dims_in);
    dog_print("filter", vec_filter, dims_filter);
    dog_print("output", vec_out, dims_out);

    dog_print("input", vec_in_grad, dims_in);
    dog_print("filter", vec_filter_grad, dims_filter);
    dog_print("output", vec_out_grad, dims_out);
    return 0;
}
