#include "common.h"
#include "wrapper.h"
#include "descriptor.h"
#include <random>
#include "global.h"
#include "conv.h"
#include "../doglib/time/timer.h"
#include "fc.h"
#include "cross_entropy.h"
#include "../../../../../usr/local/cuda/include/driver_types.h"
#include "activation.h"

using std::vector;

using dim_t = Dims;
Global global;
class A {};

void dog_print(std::string name, DeviceVector<T>& vec_vec, const dim_t& dim) {
    cout << name << endl;
    auto sz = get_volume(dim);
    assert(vec_vec.size() == sz);
    host_vector<T> vec = vec_vec;
    cudaDeviceSynchronize();
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
    std::default_random_engine e(3);
    vec_vec.resize(sz);
    if(set_value) {
        thrust::host_vector<T> host_vec(sz);
        for(auto id : Range(sz)) {
            host_vec[id] = e() % 2001 / 1000.0 - 1;
        }
        vec_vec = host_vec;
    }
}

// int workload_conv() {
//     using T = float;
//     DeviceVector<T> vec_in;
//     DeviceVector<T> vec_filter;
//     DeviceVector<T> vec_out;
//     DeviceVector<T> vec_in_grad;
//     DeviceVector<T> vec_filter_grad;
//     DeviceVector<T> vec_out_grad;
//     DeviceVector<char> vec_workspace;
//     constexpr int B = 4;
//     constexpr int Ci = 512;
//     constexpr int Co = 512;
//     constexpr int W = 64;
//     constexpr int H = 64;
//     constexpr int K = 3;
//     constexpr int group = 1;
//     constexpr int padding = 1;
//     constexpr int stride = 1;
//     constexpr int dilation = 1;
//     dim_t dims_in = {B, Ci, H, W};
//     dim_t dims_filter = {Ci, Co, K, K};
//     ConvolutionFunctor functor(dims_in, dims_filter, group, padding, stride, dilation);
//     auto dims_out = functor.get_dims_out();

//     dog_resize_to(vec_in, dims_in, true);
//     dog_resize_to(vec_in_grad, dims_in, false);
//     dog_resize_to(vec_filter, dims_filter, true);
//     dog_resize_to(vec_filter_grad, dims_filter, false);
//     dog_resize_to(vec_out, dims_out, false);
//     dog_resize_to(vec_out_grad, dims_out, true);

//     global.update_workspace_size(functor.workspace_fwd());
//     global.update_workspace_size(functor.workspace_bwd_data());
//     global.update_workspace_size(functor.workspace_bwd_filter());

//     doglib::time::TimerAdvanced timer([]() { cudaDeviceSynchronize(); });

//     functor.forward(vec_out, vec_in, vec_filter);
//     cout << timer.get_step_seconds() << endl;

//     functor.backwardData(vec_in_grad, vec_out_grad, vec_filter);
//     // cout << timer.get_step_seconds() << endl;

//     functor.backwardFilter(vec_filter_grad, vec_out_grad, vec_in);
//     cout << timer.get_step_seconds() << endl;

//     // dog_print("input", vec_in, dims_in);
//     // dog_print("filter", vec_filter, dims_filter);
//     // dog_print("output", vec_out, dims_out);

//     // dog_print("input", vec_in_grad, dims_in);
//     // dog_print("filter", vec_filter_grad, dims_filter);
//     // dog_print("output", vec_out_grad, dims_out);
//     return 0;
// }

DeviceVector<int> get_labels(const DeviceVector<T>& data, int batch, int entry_size) {
    vector<int> tmp;
    thrust::host_vector<T> h_d(data);

    for(auto bid : Range(batch)) {
        double sum = 0;
        for(auto eid : Range(entry_size)) {
            sum += h_d[bid * entry_size + eid];
        }
        tmp.push_back(sum >=  0);
    }
    return tmp;
}

struct functor
{
  __host__ __device__
  bool operator()(float x)
  {
    return x < 1;
  }
};

int main() {
    int N = 3000;
    int batch = N;
    int in_size = 10;
    int class_size = 2;
    DeviceVector<T> d_loss;
    DeviceVector<T> data;
    DeviceVector<T> parameters;
    DeviceVector<T> parameters_grad;
    DeviceVector<T> feature_map;
    DeviceVector<T> grad_map;
    FCFunctor fc(batch, in_size, class_size);
    CrossEntropy ce(N, class_size);
    global.update_workspace_size(ce.workspace());

    dog_resize_to(d_loss, {N});
    dog_resize_to(data, {N, in_size}, true);
    dog_resize_to(parameters, {(int)fc.size_parameters()}, true);
    dog_resize_to(parameters_grad, {(int)fc.size_parameters()}, true);
    dog_resize_to(feature_map, {N, class_size}, true);
    dog_resize_to(grad_map, {N, class_size}, false);
    auto labels = get_labels(data, batch, in_size);

    for(auto lb : labels) {
        cout << lb << " ";
    }
    cout << endl;
    for(auto iteration : Range(1000)) {
        float loss = 0;
        thrust::fill_n(thrust::device, parameters_grad.begin(), in_size * class_size, 0.00233);
        // dog_print("x", data, {N, in_size});
        // dog_print("Wb", parameters, {1 + in_size, class_size});
        fc.forward(feature_map, data, parameters);
        // dog_print("y", feature_map, {N, class_size});
        ce.forward(d_loss, feature_map, labels);
        // dog_print("loss", d_loss, {N});
        loss = thrust::reduce(thrust::device, d_loss.begin(), d_loss.end());
        int correct = thrust::count_if(thrust::device, d_loss.begin(), d_loss.end(), functor());
        cout <<  "^^" <<  loss / N  << "%%" << correct << endl;
        // cout <<  "^^" <<  loss / N  << endl;
        ce.backward(grad_map, d_loss, labels);
        // dog_print("@y", grad_map, {N, class_size});
        fc.backward(nullptr, parameters_grad, data, grad_map, parameters);
        // dog_print("@Wb", parameters_grad, {1 + in_size, class_size});


        thrust::transform(thrust::device, parameters.begin(), parameters.end(),
                          parameters_grad.begin(), parameters.begin(),
                          thrust::plus<float>());

        // cout << "&&&" << endl;
        // cout << "&&&" << endl;
        // cout << "&&&" << endl;
        if(loss < .0000001) break;
    }
}
