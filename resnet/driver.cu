#include "../doglib/graph/procedure.h"
#include "cg.h"
#include "components/cross_entropy.h"
#include <random>
#include <arpa/inet.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <fstream>
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

void dog_log(float* ptr, const dim_t& dim) {}
using std::vector;
const char* data_file = "/home/guilin/workspace/data/mnist/images-idx3-ubyte";
const char* labels_file = "/home/guilin/workspace/data/mnist/labels-idx1-ubyte";

host_vector<float> get_data() {
    host_vector<float> data;
    std::ifstream fin(data_file, std::ios::binary);
    uint32_t magic, number, w, h;
    fin.read((char*)&magic, 4);
    fin.read((char*)&number, 4);
    fin.read((char*)&w, 4);
    fin.read((char*)&h, 4);
    magic = htonl(magic);
    number = htonl(number);
    h = htonl(h);
    w = htonl(w);
    assert(magic == 0x00000803);
    assert(number == 60000);
    assert(h == 28);
    assert(w == 28);
    number = 60000;
    auto sz = number * w * h;
    data.resize(sz);
    vector<uint8_t> buffer(sz);
    fin.read((char*)buffer.data(), sz);
    for(auto id : Range(sz)) {
        uint8_t x = buffer[id];
        assert(0 <= x && x < 256);
        data[id] = x / 255.0;
    }
    return data;
}

host_vector<int> get_labels() {
    host_vector<int> data;
    std::ifstream fin(labels_file, std::ios::binary);
    uint32_t magic, number;
    fin.read((char*)&magic, 4);
    fin.read((char*)&number, 4);
    magic = htonl(magic);
    number = htonl(number);
    assert(magic == 0x00000801);
    assert(number == 60000);
    number = 60000;
    auto sz = number;
    data.resize(sz);
    vector<uint8_t> buffer(sz);
    fin.read((char*)buffer.data(), sz);
    for(auto id : Range(sz)) {
        uint8_t x = buffer[id];
        assert(0 <= x && x < 10);
        data[id] = x;
    }
    return data;
}

float get_acc(float* dev_logits, int* labels, int N, int feature) {
    vector<float> buffer(N * feature);
    cudaMemcpy(buffer.data(), dev_logits, N * feature * sizeof(float), cudaMemcpyDefault);
    int count = 0;
    for(int b : Range(N)) {
        auto loc = std::max_element(buffer.begin() + b * feature,
                                    buffer.begin() + (b + 1) * feature) -
                   buffer.begin() - b * feature;
        //
        assert(0 <= loc && loc < feature);
        assert(0 <= loc && loc < feature);
        count += (loc == labels[b]) ? 1 : 0;
    }
    return count * 1.0 / N;
}

Global global;
int main() {
    Engine eng;
    // define network structure
    int B = 1;
    int features = 4;
    // int hidden = 28 * 28;
    int hidden = 4;
    int classes = 2;
    dim_t input_dim = {B, features};

    auto x = eng.insert_leaf<PlaceHolderNode>(input_dim);
    eng.src_node = x;

    // auto shortcut = x;
    // x = eng.insert_node<FCNode>(x, B, features, hidden);
    // x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    // x = eng.insert_node<FCNode>(x, B, hidden, hidden);
    // x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    // x = eng.insert_node<FCNode>(x, B, hidden, hidden);
    // x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    // x = eng.insert_blend<AddNode>(x, shortcut, dim_t{B, hidden});

    x = eng.insert_node<FCNode>(x, B, hidden, classes);
    eng.dest_node = x;
    eng.finish_off();
    
    // auto total = 60000;
    // host_vector<float> data_raw = get_data();
    // host_vector<int> labels_raw = get_labels();

    auto total = B;
    host_vector<float> data_raw;
    host_vector<int> labels_raw;
    data_raw.resize(B * 1000);
    std::default_random_engine e(201);
    for(auto& x : data_raw) {
        x = (float)(e() % 10001) / 5000 - 1;
    }
    for(auto id : Range(B)) {
        float sum = 0;
        for(auto x : Range(features)) {
            sum += data_raw[id * features + x];
        }
        int label = sum >= 0 ? 1 : 0;
        labels_raw.push_back(label);
    }

    for(auto x: labels_raw){
        cout << x; 
    }  
    cout << endl;

    DeviceVector<T> losses(B);
    CrossEntropy ce(B, classes);
    global.update_workspace_size(ce.workspace());
    for(auto x : Range(3)) {
        auto offset_lb = x % (total / B) * B;
        auto offset_dt = offset_lb * features;
        auto data_beg = data_raw.data() + offset_dt;
        auto data_end = data_raw.data() + offset_dt + B * features;
        auto labels_beg = labels_raw.data() + offset_lb;
        auto labels_end = labels_raw.data() + offset_lb + B;
        eng.zero_grad();
        eng.forward_pass(data_beg);
        auto act = eng.get_ptr(eng.dest_node);
        auto act_grad = eng.get_ptr(~eng.dest_node);
        device_vector<int> dev_labels(labels_beg, labels_end);
        dog_print("##", act, dim_t{B, classes});
        ce.forward(losses, act, dev_labels.data().get());
        // eng.get_mm().l2_forward(losses, B, 0.1);
        // dog_print("??", losses, dim_t{B});
        auto loss = thrust::reduce(thrust::device, losses.begin(), losses.end());

        // eng.get_mm().l2_backward(losses, B, 0.1);
        ce.backward(act_grad, 0.1, act, losses, dev_labels.data().get());
        // dog_print("SS", act_grad, dim_t{B, classes});
        // dog_print("hhd", act, {B});

        eng.backward_pass(act_grad);
        // auto correct = thrust::count_if(losses.begin(), losses.end(), functor());
        auto correct = get_acc(act, labels_beg, B, classes);
        if(loss != loss) {
            break;
        }
        if(x % 100) {
            eng.step();
            cout << loss / B << " " << correct << endl;
        } else {
            cout << "test: " << loss / B << " " << correct << endl;
        }
    }
}