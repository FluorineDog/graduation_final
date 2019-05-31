#include "visitors.h"
#include "computational_graph.h"

void Engine::prepare_feature_maps() {
    MetaVisitor meta;
    for(auto& pr : nodes) {
        auto id = std::get<0>(pr);
        auto& node = std::get<1>(pr);
        auto map_dim = meta.analyse(*node).map_dim;
        mm.register_map(id, get_volume(map_dim));
    }
}

void Engine::prepare_gradient_maps() {
    MetaVisitor meta;
    for(auto& pr : nodes) {
        auto id = std::get<0>(pr);
        auto& node = std::get<1>(pr);
        auto map_dim = meta.analyse(*node).map_dim;
        mm.register_map(~id, get_volume(map_dim));
    }
}

void Engine::prepare_workspace() {
    MetaVisitor meta;
    for(auto& pr : nodes) {
        auto id = std::get<0>(pr);
        auto& node = std::get<1>(pr);
        auto size = meta.analyse(*node).workspace;
        global.update_workspace_size(size);
    }
}


void Engine::register_weight_maps() {
    MetaVisitor meta;
    ProcedureDFS dfs(backward_graph);
    dfs.set_visitor(Transfer::finish, [&, this](int, int id) {
        auto& node = *this->nodes[id];
        auto size = meta.weight_size(node);
        opt.register_weight(id, size);
    });
    dfs.execute_at(dest_node);
    opt.finish_weight();
}

void Engine::forward_pass(float* input) {
    ForwardVisitor fwd(*this);
    fwd.set(input);
    ProcedureDFS dfs(backward_graph);
    dfs.set_visitor(Transfer::finish, [&, this](int, int id) {
        auto& node = *this->nodes[id];
        node.accept(fwd);
    });
    dfs.execute_at(dest_node);
}

void Engine::backward_pass(float* act_grad) {
    BackwardVisitor bwd(*this);
    auto dim = MetaVisitor().out_dim(*nodes[dest_node]);
    auto top = mm.get(~dest_node);
    cudaMemcpy(top, act_grad, get_volume(dim) * sizeof(float), cudaMemcpyDefault);
    ProcedureDFS dfs(forward_graph);
    dfs.set_visitor(Transfer::finish, [&, this](int, int id){
        auto& node = *nodes[id];
        node.accept(bwd);
    });
    dfs.execute_at(src_node);
}
