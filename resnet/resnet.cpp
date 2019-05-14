#include "common.h"
#include "wrapper.h"
#include "descriptor.h"
#include "global.h"
#include "conv.h"
#include "../doglib/time/timer.h"
#include "fc.h"
#include "cross_entropy.h"
#include "../../../../../usr/local/cuda/include/driver_types.h"

using std::vector;

using dim_t = Dims;
Global global;
class A {};

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
            host_vec[id] = (T)(id % 257 / 128.0);
        }
        vec_vec = host_vec;
    }
}

int workload_conv() {
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

    doglib::time::TimerAdvanced timer([]() { cudaDeviceSynchronize(); });

    functor.forward(vec_out, vec_in, vec_filter);
    cout << timer.get_step_seconds() << endl;

    functor.backwardData(vec_in_grad, vec_out_grad, vec_filter);
    // cout << timer.get_step_seconds() << endl;

    functor.backwardFilter(vec_filter_grad, vec_out_grad, vec_in);
    cout << timer.get_step_seconds() << endl;

    // dog_print("input", vec_in, dims_in);
    // dog_print("filter", vec_filter, dims_filter);
    // dog_print("output", vec_out, dims_out);

    // dog_print("input", vec_in_grad, dims_in);
    // dog_print("filter", vec_filter_grad, dims_filter);
    // dog_print("output", vec_out_grad, dims_out);
    return 0;
}

DeviceVector<int> get_labels(const DeviceVector<T>& data, int batch, int entry_size) {
    vector<int> tmp; thrust::host_vector<T> h_d(data);

    for(auto bid : Range(batch)) {
        double sum = 0;
        for(auto eid : Range(entry_size)) {
            sum += h_d[bid * entry_size + eid];
        }
        tmp.push_back(sum >= entry_size);
    }
    return tmp;
}

int main() {
    int N = 256;
    int batch = N;
    int in_size = 128;
    int class_size = 2;
    DeviceVector<T> d_loss;
    DeviceVector<T> data;
    DeviceVector<T> parameters;
    DeviceVector<T> parameters_grad;
    DeviceVector<T> feature_map;
    DeviceVector<T> grad_map;
    FCFunctor fc(batch, in_size, class_size);
    CrossEntropy ce(class_size, N);
    global.update_workspace_size(ce.workspace());

    dog_resize_to(d_loss, {1}, true);
    dog_resize_to(data, {N, in_size}, true);
    dog_resize_to(parameters, {(int)fc.size_parameters()}, true);
    dog_resize_to(parameters_grad, {(int)fc.size_parameters()}, true);
    dog_resize_to(feature_map, {N, class_size}, false);
    dog_resize_to(grad_map, {N, class_size}, false);
    auto labels = get_labels(data, batch, class_size);
    for(auto iteration : Range(1)) {
        cout << grad_map.size() << endl;
        thrust::fill_n(thrust::device, grad_map.begin(), batch * class_size, 0.0f);
        fc.forward(feature_map, data, parameters);
        ce.forward(d_loss, feature_map, labels);
        float loss;
        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        // float loss_grad = -0.01f * loss / N;
        // ce.backward(grad_map, loss_grad, labels);
        // fc.backward(nullptr, parameters_grad, data, grad_map, parameters);
        cout << loss << endl;
    }
}
