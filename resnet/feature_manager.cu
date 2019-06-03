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

    // generate dual graph
    DynamicGraph dual_graph(N);
    for(auto from : Range(N)) {
        for(auto to : graph.adjacent(from)) {
            dual_graph.add_edge(from, to);
            dual_graph.add_edge(to, from);
        }
    }

    ProcedureDFS dfs(dual_graph);
    int discover_time = 0;
    dfs.set_visitor(Transfer::discover, [&](int u, int v) {
        meta[v].parent = u;
        auto dt = discover_time++;
        meta[v].id = dt;
        meta[v].low = dt;
    });
    dfs.set_visitor(Transfer::revisit_processing, [&](int u, int v) {
        assert(u != -1);
        if(meta[u].parent == v) {
            return;
        }
        meta[u].low = std::min(meta[u].low, meta[v].low);
    });
    dfs.set_visitor(Transfer::finish, [&](int u, int v) {
        if(u == -1) {
            return;
        }
        meta[u].low = std::min(meta[u].low, meta[v].low);
        // v is fixed
        if(meta[u].id <= meta[v].low) {
            meta[u].articulation = true;
        }
    });
    dfs.execute_at(root);
    vector<bool> result(N);
    for(auto i : Range(N)) {
        result[i] = meta[i].articulation;
    }
    result[root] = true;
    return result;
}

auto choose_breakpoints(const DynamicGraph& graph, const vector<bool>& is_breakings) {
    int N = graph.n_vertex();
    auto orders = toposort_cycle(graph);
    std::reverse(orders.begin(), orders.end());
    int M = 0;
    auto cur_iter = orders.begin();
    std::set<int> pivots;
    pivots.insert(orders.back());
    pivots.insert(orders.front());
    auto output_iter = orders.end() - 1;
    while(cur_iter != output_iter) {
        auto iter = cur_iter + 1;
        // find the closest breakpoint
        while(iter < output_iter && !is_breakings[*iter]) {
            ++iter;
        }
        if(iter - cur_iter >= M) {
            pivots.insert(*iter);
            M = iter - cur_iter + 1;
            cur_iter = iter;
            continue;
        }
        auto valid_iter = iter;
        while(iter <= output_iter && iter - cur_iter <= M) {
            if(is_breakings[*iter]) {
                valid_iter = iter;
            }
            ++iter;
        }
        pivots.insert(*valid_iter);
        M = M + 1;
        cur_iter = valid_iter;
    }
    // generate execution plan
    // forward plan
    // how can i do so?
    return pivots;
}

enum class ExecType {
    forward,
    backward,
    free_feature    // and
};

struct ExecPlan {
    ExecType type;
    int node_id;
    ExecPlan(ExecType type, int node_id) : type(type), node_id(node_id) {}
};
using std::set;
using std::vector;
auto gen_forward_plan(const DynamicGraph& forward_graph, const set<int>& brkpnts) {
    //
    auto orders = toposort_acycle(forward_graph);
    auto backward_graph = transpose(forward_graph);
    int N = forward_graph.n_vertex();
    vector<int> ref_counts(N);
    for(auto from : Range(N)) {
        int ref = 0;
        for(auto to : forward_graph.adjacent(from)) {
            ++ref;
        }
        ref_counts[from] = ref;
    }
    vector<ExecPlan> plans;
    for(auto v : orders) {
        plans.emplace_back(ExecType::forward, v);
        for(auto par : backward_graph.adjacent(v)) {
            auto ref = --ref_counts[par];
            if(ref == 0 && brkpnts.count(par) == 0) {
                plans.emplace_back(ExecType::free_feature, par);
            }
        }
    }
    for(auto x: ref_counts){
        assert(x == 0);
    }
    return plans;
}

auto gen_backward_plan(const DynamicGraph& forward_graph, const set<int>& brkpnts) {
    auto orders = toposort_acycle(forward_graph);
    auto rev_orders = orders;
    auto N = forward_graph.n_vertex();
    std::reverse(rev_orders.begin(), rev_orders.end());

    ExecPlan plans;
    vector<char> featured(N);
    for(auto id: brkpnts) {
        featured[id] = true; 
    }

    for(auto v: rev_orders) {
        if(featured[v]) {
            // recover plan
        }        
         
    }
}

void FeatureManager::analyse() {
    if(false) {
        // step 1: find all breakpoints
        // step 2: choose the breakpoints
        // step 3: generate forward execution plan
        // step 4: generate backwawrd execution plan
        // step 5: execute it
        // this is hard, but i believe you can do all
    }
    // step 1: find all breakpoints
    auto candidates = find_breakpoints(eng.forward_graph, eng.src_node);
    candidates[eng.dest_node] = true;
    // step 2: choose the breakpoints
    auto brkpnts = choose_breakpoints(eng.forward_graph, candidates);
    // step 3: generate execution plan
    auto fwd_plan = gen_forward_plan(eng.forward_graph, brkpnts);

    return;
}