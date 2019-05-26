#pragma once
#include "helper/defs.h"
#include "helper/common.h"
inline void show_weight(std::string name, device_vector<float>& wtf) {
    int N = 28 * 28;
    int K = 20;
    int M = 10;
    assert(wtf.size() == N * K + K + (K * K) + K + K * M + M);
    host_vector<float> hoster = wtf;
    float* ref = hoster.data();
    // skip
    ref += N * K + K;
    dog_print(name + "W", ref, {K, K});
    ref += K * K;
    dog_print(name + "b", ref, {K});
    ref += K;
    dog_print(name + "W2", ref, {K, M});
    ref += K * M;
    dog_print(name + "b2", ref, {M});
    ref += M;
    cout << endl << "......................." << endl;
}

// template <class T>
inline void dog_resize_to(device_vector<float>& vec_vec, const dim_t& dim,
                          bool set_value = false) {
    auto sz = get_volume(dim);
    std::normal_distribution<double> distribution(0, 0.1);
    std::default_random_engine e(3);
    vec_vec.resize(sz);
    thrust::host_vector<float> host_vec(sz, 0);
    if(set_value) {
        for(auto id : Range(sz)) {
            host_vec[id] = distribution(e);
        }
    }
    vec_vec = host_vec;
}

inline DeviceVector<int> get_labels(const DeviceVector<T>& data, int batch,
                                    int entry_size) {
    std::vector<int> tmp;
    thrust::host_vector<T> h_d(data);

    for(auto bid : Range(batch)) {
        double sum = 0;
        for(auto eid : Range(entry_size)) {
            sum += h_d[bid * entry_size + eid];
        }
        tmp.push_back(sum >= 0);
    }
    return tmp;
}

using namespace doglib::graph;

struct OP1 {
    __host__ __device__ float operator()(float a, float b) {
        float x = 0.9 * a + b;
        return x;
    }
};

struct OP2 {
    OP2(float x) {
        coef = x;
    }
    __host__ __device__ float operator()(float a, float b) {
        // return a - 0.01 * a * a * a + b;
        return a + b * coef;
    }
    float coef;
};

class Optimizer {
  public:
    void register_weight(int id, int size) {
        weight_offsets[id] = total_weight;
        total_weight += size;
    }
    float* get_weight(int id) {
        assert(weight.size() == total_weight);
        auto offset = weight_offsets[id];
        return weight.data().get() + offset;
    }
    float* get_weight_grad(int id) {
        assert(weight_grad.size() == total_weight);
        auto offset = weight_offsets[id];
        return weight_grad.data().get() + offset;
    }

    void finish_weight() {
        dog_resize_to(weight, {(int)total_weight}, true);
        dog_resize_to(weight_grad, {(int)total_weight}, false);
        weight_acc.resize(total_weight, 0);
        assert(weight.size() == total_weight);
        assert(weight.size() == total_weight);
    }

    void zero_grad() {
        thrust::fill(thrust::device, weight_grad.begin(), weight_grad.end(), 0);
        thrust::fill(thrust::device, weight_grad.begin(), weight_grad.end(), 0);
    }
    

    void step(float coef) {
        thrust::transform(weight_acc.begin(), weight_acc.end(), weight_grad.begin(),
                          weight_acc.begin(), OP1());
        thrust::transform(weight.begin(), weight.end(), weight_acc.begin(),
                          weight.begin(), OP2(-coef));
        // show_weight("", weight);
        // show_weight("@", weight_grad);
        // thrust::transform(weight.begin(), weight.end(), weight_grad.begin(),
        //                   weight.begin(), OP2(-coef));
    }

  private:
    std::map<int, size_t> weight_offsets;
    size_t total_weight = 0;
    device_vector<float> weight;
    device_vector<float> weight_grad;
    device_vector<float> weight_acc;
};