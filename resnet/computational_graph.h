#pragma once
#include "helper/defs.h"
#include "helper/common.h"
#include "../doglib/graph/graph.h"
#include "../doglib/graph/procedure.h"
#include <random>
// template <class T>
inline void dog_resize_to(device_vector<float>& vec_vec, const dim_t& dim,
                          bool set_value = false) {
    auto sz = get_volume(dim);
    std::default_random_engine e(3);
    vec_vec.resize(sz);
    thrust::host_vector<float> host_vec(sz, 0);
    if(set_value) {
        for(auto id : Range(sz)) {
            host_vec[id] = e() % 2001 / 1000.0 - 1;
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

//  stupid version
//  definition:
//    accept correct
class MemoryManager {
  public:
    void init() {
        //
    }
    void register_map(int id, size_t size) {
        assert(!mapping.count(id));
        mapping[id].resize(size);
    }
    float* get(int id) {
        assert(mapping.count(id));
        return mapping[id];
    }
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
        assert(weight.size() == total_weight);
        assert(weight.size() == total_weight);
    }
    void free(int id) {}
    void terminate() {
        mapping.clear();
    }
    void zero_grad() {
        for(auto pr : mapping) {
            auto& vec = pr.second;
            thrust::fill(vec.begin(), vec.end(), 0);
        }
        thrust::fill(weight_grad.begin(), weight_grad.end(), 0);
    }
    void step() {
        thrust::transform(weight.begin(), weight.end(), weight_grad.begin(),
                          weight.begin(), thrust::plus<float>());
    }

  private:
    std::map<int, DeviceVector<float>> mapping;
    std::map<int, size_t> weight_offsets;
    size_t total_weight = 0;
    device_vector<float> weight;
    device_vector<float> weight_grad;
};

// graph executor, in one place
class Engine {
  public:
    Engine() : forward_graph(0), backward_graph(0) {
        // src_node and dest_node is here waiting
    }

    void prepare_feature_maps();
    void prepare_gradient_maps();    // (todo)
    void register_weight_maps();     //: better with hashtable

    void finish_off() {
        backward_graph = transpose(forward_graph);
        prepare_feature_maps();
        prepare_gradient_maps();    // (todo)
        register_weight_maps();     //: better with hashtable
    }
    template <class T, class... Arg>
    int insert_leaf(Arg... args) {
        auto id = forward_graph.new_vertex();
        nodes.emplace(id, std::make_unique<T>(id, args...));
        return id;
    }

    template <class T, class... Arg>
    int insert_node(int parent, Arg... args) {
        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(parent, id);
        nodes.emplace(id, std::make_unique<T>(parent, id, args...));
        return id;
    }

    template <class T, class... Arg>
    int insert_blend(int a, int b, Arg... args) {
        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(a, id);
        forward_graph.add_edge(b, id);
        nodes.emplace(id, std::make_unique<T>(a, b, id, args...));
        return id;
    }

    void forward_pass(float* input);
    void backward_pass(float* logits_grad);
    void zero_grad() {
        mm.zero_grad();
    }

    void step() {
        mm.step();
    }

    MemoryManager& get_mm() {
        return mm;
    }

    float* get_ptr(int id) {
        return mm.get(id);
    }

    int dest_node;
    int src_node;

    DynamicGraph forward_graph;
    DynamicGraph backward_graph;
    std::map<int, std::unique_ptr<NodeBase>> nodes;
    MemoryManager mm;
};

enum class Mode { Collect, Optimized };
