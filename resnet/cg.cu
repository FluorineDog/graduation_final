#include "cg.h"

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

void Engine::register_weight_maps() {
    MetaVisitor meta;
    ProcedureDFS dfs(backward_graph);
    dfs.set_visitor(Transfer::finish, [&, this](int, int id) {
        auto& node = *this->nodes[id];
        auto size = meta.weight_size(node);
        mm.register_weight(id, size);
    });
}


void Engine::forward_pass(float* input) {
    ForwardVisitor fwd(*this);
    fwd.set(input);
    ProcedureDFS dfs(backward_graph);
    dfs.set_visitor(Transfer::finish, [&, this](int, int id){
        auto& node = *this->nodes[id];
        node.accept(fwd); 
    });
}
