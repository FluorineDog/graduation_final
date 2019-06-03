#include "helper/common.h"
#include "visitors.h"
#include "engine.h"

void Engine::prepare_feature_maps() {
    MetaVisitor meta;
    for(auto& pr : nodes) {
        auto id = std::get<0>(pr);
        auto& node = std::get<1>(pr);
        auto map_dim = meta.analyse(*node).map_dim;
        mm.register_feature_map(id, get_volume(map_dim));
    }
}

void Engine::prepare_gradient_maps() {
    MetaVisitor meta;
    for(auto& pr : nodes) {
        auto id = std::get<0>(pr);
        auto& node = std::get<1>(pr);
        auto map_dim = meta.analyse(*node).map_dim;
        mm.register_gradient_map(id, get_volume(map_dim));
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
    ForwardVisitor fwd_v(*this);
    fwd_v.set(input);
    for(auto plan : fwd_plan_) {
        auto& node = *nodes[plan.node_id];
        switch(plan.type) {
            case ExecType::backward: {
                // node.accept(fwd);
                assert(false);
                break;
            }
            case ExecType::forward: {
                node.accept(fwd_v);        
                break;
            }
            case ExecType::free_feature: {
                // silence
                // mm.free_feature(plan.node_id); 
                break;
            }
            default: break;
        }
    }
    // ProcedureDFS dfs(backward_graph);
    // dfs.set_visitor(Transfer::finish, [&, this](int, int id) {
    //     auto& node = *this->nodes[id];
    //     node.accept(fwd);
    // });
    // dfs.execute_at(dest_node);
}

void Engine::backward_pass(float* act_grad) {
    ForwardVisitor forwd(*this);
    BackwardVisitor bwd(*this);
    auto dim = MetaVisitor().out_dim(*nodes[dest_node]);
    auto top = mm.get_gradient(dest_node);
    cudaMemcpy(top, act_grad, get_volume(dim) * sizeof(float), cudaMemcpyDefault);

    for(auto plan : bwd_plan_) {
        auto& node = *nodes[plan.node_id];
        switch(plan.type) {
            case ExecType::backward: {
                node.accept(bwd);
                break;
            }
            case ExecType::forward: {
                // silence
                // node.accept(forwd); 
                break;
            }
            case ExecType::free_feature: {
                // mm.free_feature(plan.node_id); 
                break;
            }
            default: break;
        }
    }
    // ProcedureDFS dfs(forward_graph);
    // dfs.set_visitor(Transfer::finish, [&, this](int, int id){
    //     auto& node = *nodes[id];
    //     node.accept(bwd);
    // });
    // dfs.execute_at(src_node);
}
