#include "../doglib/graph/procedure.h"
#include "cg.h"
#include "components/cross_entropy.h"
#include <random>

#include <thrust/count.h>
#include <thrust/device_vector.h>
struct functor {
    __host__ __device__ bool operator()(float x) {
        return x < 0.6931471805599453;
    }
};

// void dog_print(std::string name, const float* ptr, const dim_t& dim) {
//     cout << name << endl;
//     auto sz = get_volume(dim);

//     cudaDeviceSynchronize();
//     host_vector<T> vec(sz);
//     cudaMemcpy(vec.data(), ptr, sz * sizeof(float), cudaMemcpyDefault);
//     auto tmp = dim;
//     std::reverse(tmp.begin(), tmp.end());
//     for(auto index : Range(sz)) {
//         int index_cpy = index;
//         for(auto x : tmp) {
//             if(index_cpy % x != 0) break;
//             index_cpy /= x;
//             cout << "--------" << endl;
//         }
//         cout << vec[index] << " ";
//     }
//     cout << endl << "##########" << endl;
// }

void dog_log(float* ptr, const dim_t& dim){

}

Global global;
int main() {
    Engine eng;
    // define network structure
    int B = 10000;
    int features = 8;
    int hidden = features;
    int classes = 2;
    dim_t input_dim = {B, features};

    auto x = eng.insert_leaf<PlaceHolderNode>(input_dim);
    eng.src_node = x;
    auto shortcut = x;
    x = eng.insert_node<FCNode>(x, B, features, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_node<FCNode>(x, B, hidden, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_blend<AddNode>(x, shortcut, dim_t{B, hidden});
    x = eng.insert_node<FCNode>(x, B, hidden, classes);
    eng.dest_node = x;
    eng.finish_off();

    host_vector<float> input;
    input.resize(B * 1000);
    std::default_random_engine e(201);
    for(auto& x : input) {
        x = (float)(e() % 10001) / 5000 - 1;
    }

    host_vector<int> labels;
    for(auto id : Range(B)) {
        float sum = 0;
        for(auto x : Range(features)) {
            sum += input[id * features + x];
        }
        int label = sum >= 0 ? 1 : 0;
        labels.push_back(label);
    }

    for(auto x : labels) {
        cout << x << " ";
    }
    cout << endl;

    device_vector<int> dev_labels = labels;
    DeviceVector<T> losses(B);
    CrossEntropy ce(B, classes);
    global.update_workspace_size(ce.workspace());
    for(auto x : Range(100)) {
        eng.zero_grad();
        eng.forward_pass(input.data());
        auto act = eng.get_ptr(eng.dest_node);
        auto act_grad = eng.get_ptr(~eng.dest_node);

        ce.forward(losses, act, dev_labels.data().get());
        // dog_print("##", act, dim_t{B, classes});
        auto loss = thrust::reduce(thrust::device, losses.begin(), losses.end());
        ce.backward(act_grad, 0.0001, losses, dev_labels.data().get());
        // dog_print("SS", act_grad, dim_t{B, classes});
        // // dog_print("hhd", act, {B});
        
        eng.backward_pass(act_grad);
        int correct = thrust::count_if(losses.begin(), losses.end(), functor());
        eng.step();
        cout << "^^" << loss / B << "%%" << correct << endl;
    }
}