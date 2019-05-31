#pragma once
#include "helper/defs.h"
#include "helper/common.h"
#include "../doglib/graph/graph.h"
#include "../doglib/graph/procedure.h"
#include <random>
#include "optimizer.h"
#include "visitors.h"
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
    void zero_grad() {
        for(auto& pr : mapping) {
            auto& vec = pr.second;
            auto ptr = vec.data().get();
            thrust::fill(thrust::device, vec.begin(), vec.end(), 0);
            // assert(inspect(ptr) == 0);
        }
    }

    void free(int id) {}
    void terminate() {
        mapping.clear();
    }

  private:
    std::map<int, DeviceVector<float>> mapping;
};

// graph executor, in one place
class Engine {
  public:
    Engine() : forward_graph(0), backward_graph(0) {
        // src_node and dest_node is here waiting
    }

    void prepare_feature_maps();
    void prepare_workspace();
    void prepare_gradient_maps();    // (todo)
    void register_weight_maps();     //: better with hashtable

    void finish_off() {
        backward_graph = transpose(forward_graph);
        prepare_feature_maps();
        prepare_workspace();
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
    int insert_node(int parent, dim_t in_dims, Arg... args) {
        MetaVisitor meta;
        auto ref_dims = meta.out_dim(*nodes[parent]);
        assert(ref_dims.size() == in_dims.size());
        for(auto i: Range(ref_dims.size())){
            assert(ref_dims[i] == in_dims[i]);
        }
        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(parent, id);
        
        nodes.emplace(id, std::make_unique<T>(parent, id, in_dims, args...));
        return id;
    }

    template <class T, class... Arg>
    int insert_blend(int a, int b, dim_t in_dims, Arg... args) {
        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(a, id);
        forward_graph.add_edge(b, id);
        nodes.emplace(id, std::make_unique<T>(a, b, id, in_dims, args...));
        return id;
    }

    void forward_pass(float* input);
    void backward_pass(float* logits_grad);
    void zero_grad() {
        mm.zero_grad();
        opt.zero_grad();
    }

    MemoryManager& get_mm() {
        return mm;
    }
    Optimizer& get_opt() {
        return opt;
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
    Optimizer opt;
};

enum class Mode { Collect, Optimized };
