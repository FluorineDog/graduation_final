#pragma once
#include "defs.h"
#include "common.h"
#include "../doglib/graph/graph.h"
#include "../doglib/graph/procedure.h"
#include <optional>

using namespace doglib::graph;


//  stupid version
//  definition:
//    accept correct
class MemoryManager {
  public:
    void init() {
        //
    }
    float* prepare(int id, size_t size) {
        if(!mapping.count(id)) {
            mapping[id].resize(size);
        }
        return mapping[id];
    }
    float* get(int id) {
        assert(mapping.count(id));
        return mapping[id];
    }
    float* register_weight(int id, size_t size) {
        assert(weights.count(id) == 0);
        weights[id].resize(size);
    }
    float* get_weight(int id){
        assert(weights.count(id));
        return weights[id];
    }
    void finish(int id) {
        // do nothing
    }
    void free(int id) {}
    void terminate() {
        mapping.clear();
    }

  private:
    std::map<int, DeviceVector<float>> mapping;
    std::map<int, DeviceVector<float>> weights;
};

// graph executor, in one place
class Engine {
  public:
    Engine() : forward_graph(0), backward_graph(0) {
        // src_node and dest_node is here waiting
    }

    void define_net() {
        int B = 128;
        dim_t input_dim = {B, 1000};
        auto x = insert_leaf<PlaceHolderNode>();
        this->input_node = 0;
        x = this->insert_node<FCNode>(x, B, 1000, 1000);
        x = this->insert_node<ActivationNode>(x, dim_t{B, 1000});
        x = this->insert_node<FCNode>(x, B, 1000, 1000);
        x = this->insert_node<ActivationNode>(x, dim_t{B, 1000});
        x = this->insert_blend<FCNode>(x, B, 1)
        // nodes.emplace(x, std::make_unique<PlaceHolderNode>(x));
        // x = insert_node(x);
        
    }

    template<class T, class ... Arg>  
    int insert_leaf(Arg... args) {
        auto id = forward_graph.new_vertex();
        nodes.emplace(id, std::make_unique<T>(id, args...));
        return id;
    }

    template<class T, class ... Arg>  
    int insert_node(int parent, Arg... args){
        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(parent, id);
        nodes.emplace(id, std::make_unique<T>(parent, id, args...));
        return id;
    }

    template<class T, class ... Arg>  
    int insert_blend(int a, int b, Arg... args){
        auto id = forward_graph.new_vertex();
        forward_graph.add_edge(a, id);
        forward_graph.add_edge(b, id);
        nodes.emplace(id, std::make_unique<T>(a, b, id, args...));
        return id;
    }

    void forward_pass(){
    
    }

    void finish_off(int dest) {
        dest_node = dest;
        backward_graph = transpose(forward_graph);
    }
    MemoryManager& get_mm(){
        return mm;
    }
    
    int dest_node;
    int input_node;
    
    DynamicGraph forward_graph;
    DynamicGraph backward_graph;
    std::map<int, std::unique_ptr<NodeBase>> nodes;
    MemoryManager mm;
};

enum class Mode { Collect, Optimized };
