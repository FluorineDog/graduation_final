#include "memory_manager.h"
#include "engine.h"
using std::vector;

vector<bool> find_breakpoints(const DynamicGraph& graph, int root) {
    struct NodeMeta {
        int parent;
        int low;
        int id;
        bool articulation = false;
        int max_child;
    };
    int N = graph.n_vertex();
    vector<NodeMeta> meta(N);
    DynamicGraph dual_graph(N);
    for(auto from : Range(N)) {
        for(auto to : graph.adjacent(from)) {
            dual_graph.add_edge(from, to);
            dual_graph.add_edge(to, from);
        }
    }
    ProcedureDFS dfs(dual_graph);
    int discover_time = 0;
    dfs.set_visitor(Transfer::discover, [&](int u, int v){
        meta[v].parent = u;
        auto dt = discover_time++;
        meta[v].id = dt;   
        meta[v].low = dt;
    });
    dfs.set_visitor(Transfer::revisit_processing, [&](int u, int v){
        assert(u != -1);
        if(meta[u].parent == v) {
            return;
        }
        meta[u].low = std::min(meta[u].low, meta[v].low);
    });
    dfs.set_visitor(Transfer::finish, [&](int u, int v){
        if(u == -1){
            return;
        }
        meta[u].low = std::min(meta[u].low, meta[v].low);
        // v is fixed
        if(meta[u].id <= meta[v].low){
            meta[u].articulation = true;
        }
    });
    dfs.execute_at(root);
    vector<bool> result(N);
    for(auto i: Range(N)){
        result[i] = meta[i].articulation;
    }
    return result;
}

void FeatureManager::analyse() {
    vector<int> discover_time;
    {
        // step 1: find all breakpoints
        // step 2: choose the breakpoints
        // step 3: generate execution plan
        // step 4: map execution plan to data
        // step 5: forward using refcount
        // step 6: map execution with backward
        // this is hard, but i believe you can do all
    }    // step 1: find all breakpoints
    {
        // generate dual graph
    }
}