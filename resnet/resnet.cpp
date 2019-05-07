#include "doglib/common/common.h"
#include <thrust/device_vector.h>
#include <cudnn.h>
using namespace doglib::common;
using ull = long long;
using dim_t = std::vector<ull>;
using namespace thrust;

ull get_volume(const dim_t& vec) {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<ull>());
}

template <class T>
void dog_resize_to(device_vector<T>& dev_vec, const dim_t& dim, bool set_value = false) {
    //
    auto sz = get_volume(dim);
    dev_vec.resize(sz);
    if(set_value) {
        thrust::host_vector<T> host_vec(sz);
        for(auto id : Range(x)) {
            host_vec[id] = (T)id;
        }
        // i love it
        dev_vec = host_vec;
    }
}

int main() {
    using T = float;
    thrust::device_vector<T> dev_input;
    thrust::device_vector<T> dev_filter;
    thrust::device_vector<T> dev_output;
    thrust::device_vector<char> dev_workspace;
    // thrust::host_vector<T> host_weight;
    // thrust::host_vector<T> host_input;
    // thrust::host_vector<T> host_output;
    constexpr ull B = 1;
    constexpr ull Ci = 32;
    constexpr ull Co = 32;
    constexpr ull W = 32;
    constexpr ull H = 32;
    constexpr ull K = 3;
    dim_t dims_input = {B, Ci, W, H};
    dim_t dims_filter = {Ci, Co, K, K};
    dim_t dims_output = {B, Co, W, H};
    cudnnTensorDescriptor_t desc_in;
    cudnnFilterDescriptor_t desc_filter;
    cudnnTensorDescriptor_t desc_out;
    cudnnConvolutionDescriptor_t desc_conv;

    dog_resize_to(dev_input, dims_input, true);
    dog_resize_to(dev_filter, dims_filter, true);
    dog_resize_to(dev_output, dims_output, false);
    
     
    

    return 0;
}