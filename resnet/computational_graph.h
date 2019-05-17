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
    void finish(int id) {
        // do nothing
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
    Engine() : forward_graph(1), backward_graph(0) {
        // src_node and dest_node is here waiting
    }

    int insert(int parent, NodeBase&& functor) {
        int res_id = forward_graph.new_vertex();
        forward_graph.add_edge(parent, res_id);
        functors.emplace(res_id, std::move(functor));
        return res_id;
    }
    
    int blend_add(int a, int b) {
        int res_id = forward_graph.new_vertex();
        forward_graph.add_edge(a, res_id);
        forward_graph.add_edge(b, res_id);
    }
    void finish_off(int dest) {
        dest_node = dest;
        backward_graph = transpose(forward_graph);
    }
    MemoryManager& get_mm(){
        return mm;
    }
    static constexpr int src_node = 0;
    int dest_node = -1;

    DynamicGraph forward_graph;
    DynamicGraph backward_graph;
    std::map<int, class NodeBase> functors;
    MemoryManager mm;
};

enum class Mode { Collect, Optimized };
