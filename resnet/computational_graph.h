#include "common.h"
#include "../doglib/graph/graph.h"
#include "../doglib/graph/procedure.h"
#include "functor.h"

using namespace doglib::graph;

// the graph is just a driver, execute it!
class ExecutorBase{
  public: 
    virtual void execute();
};





class Engine {
  public:
    Engine() : forward_graph(1), backward_graph(0), functors(1, FunctorSpecial()) {
        // src_node and dest_node is here waiting
    }
    
    int insert(int parent, FunctorBase&& functor) {
        int res_id = forward_graph.new_vertex();
        forward_graph.add_edge(parent, res_id);
        functors.emplace_back(std::move(functor));
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
    void forward(float* logits, const float* input) {
         
    }
    static constexpr int src_node = 0;
    int dest_node = -1;

  private:
    DynamicGraph forward_graph;
    DynamicGraph backward_graph;
    std::vector<FunctorBase> functors;
};

enum class Mode { Collect, Optimized };

// stupid version
class MemoryManager {
  public:
    void init() {}
    float* prepare(int id, size_t size) {
        if(!mapping.count(id)) {
            mapping[id].resize(size);
        }
        return mapping[id];
    }
    void unused(int id) {
        // do nothing 
    }
    void free(int id) {}
    void terminate() {
        mapping.clear();
    }

  private:
    std::map<int, DeviceVector<float>> mapping;
};