#include "memory_manager.h"

void GradientManager::register_gradient_map(int node_id, size_t size) {
    assert(node_id >= 0);
    if(node_id <= meta_.size()) {
        meta_.resize(node_id + 1, 0);
        reference_.resize(node_id + 1, std::make_pair(-1, nullptr));
    }
    assert(meta_[node_id] == 0);
    meta_[node_id] = size;
}

float* GradientManager::get_gradient(int node_id) {
    assert(node_id >= 0);
    auto& ref = reference_[node_id];
    if(ref.second) {
        assert(slots_[ref.first].data().get() == ref.second);
        assert(meta_[node_id] <= slots_[ref.first].size());
        return ref.second;
    }
    assert(ref.first == -1);
    if(free_list_.empty()) {
        auto slot_id = slots_.size();
        auto sz = meta_[node_id];
        slots_.emplace_back(sz, 0);
        ref = std::make_pair(slot_id, slots_[slot_id].data().get());
        assert(slots_[ref.first].data().get() == ref.second);
        assert(meta_[node_id] <= slots_[ref.first].size());
        return ref.second;
    }
    auto slot_id = free_list_.top();
    free_list_.pop();
    auto sz = meta_[node_id];
    auto& vec = slots_[slot_id];
    if(sz >= vec.size()) {
        vec.clear();
        vec.resize(sz);
    }
    thrust::fill(vec.begin(), vec.end(), 0);
    ref = std::make_pair(slot_id, vec.data().get());

    assert(slots_[ref.first].data().get() == ref.second);
    assert(meta_[node_id] <= slots_[ref.first].size());
    return ref.second;
}

GradientDataHolder GradientManager::get_gradient_final(int node_id) {
    assert(node_id >= 0);
    auto ptr = reference_[node_id].second;
    assert(ptr);
    assert(ptr == slots_[reference_[node_id].first].data().get());
    assert(meta_[node_id] <= slots_[reference_[node_id].first].size());
    return GradientDataHolder(*this, node_id, ptr);
}

GradientDataHolder::~GradientDataHolder() {
    if(node_id < 0) {
        return;
    }
    auto ref = gm.reference_[node_id];
    gm.free_list_.push(ref.first);
    ref = std::make_pair(-1, nullptr);
}