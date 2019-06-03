#pragma once
#include "memory_manager.h"
#include "../doglib/graph/graph.h"
#include "../doglib/graph/procedure.h"
#include <random>
#include "optimizer.h"
#include "visitors.h"



// graph executor, in one place
class Engine {
  public:
    Engine() : forward_graph(0), backward_graph(0), mm() {
        // src_node and dest_node is here waiting
    }
    friend class FeatureManager;
    void prepare_feature_maps();
    void prepare_workspace();
    void prepare_gradient_maps();    // (todo)
    void register_weight_maps();     

    void finish_off() {
        backward_graph = transpose(forward_graph);
        prepare_feature_maps();
        prepare_workspace();
        prepare_gradient_maps();    // (todo)
        register_weight_maps();    
        std::tie(fwd_plan_, bwd_plan_) = mm.analyse(forward_graph, src_node, dest_node);
    }
    template <class T, class... Arg>
    int insert_leaf(Arg... args) {
        auto id = forward_graph.new_vertex();
        nodes.emplace(id, std::make_unique<T>(id, args...));
        return id;
    }

    template <class T, class... Arg>
    int insert_node(int parent, Arg... args) {
        MetaVisitor meta;
        auto in_dims = meta.out_dim(*nodes[parent]);

        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(parent, id);
        
        nodes.emplace(id, std::make_unique<T>(parent, id, in_dims, args...));
        return id;
    }

    template <class T, class... Arg>
    int insert_blend(int a, int b, Arg... args) {
        MetaVisitor meta;
        auto a_dims = meta.out_dim(*nodes[a]);
        auto b_dims = meta.out_dim(*nodes[b]);
        assert(a_dims.size() == b_dims.size());
        for(auto i: Range(a_dims.size())){
            assert(a_dims[i] == b_dims[i]);
        }

        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(a, id);
        forward_graph.add_edge(b, id);
        nodes.emplace(id, std::make_unique<T>(a, b, id, a_dims, args...));
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

    auto get_dest_feature() {
        return mm.get_feature(dest_node);
    }
    auto get_dest_gradient() {
        return mm.get_gradient(dest_node);
    }

    int dest_node;
    int src_node;

    DynamicGraph forward_graph;
    DynamicGraph backward_graph;
    vector<ExecPlan> fwd_plan_;
    vector<ExecPlan> bwd_plan_;
    std::map<int, std::unique_ptr<NodeBase>> nodes;
    MemoryManager mm;
    Optimizer opt;
};
