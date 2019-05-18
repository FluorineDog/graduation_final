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
