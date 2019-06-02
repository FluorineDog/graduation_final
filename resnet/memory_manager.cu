#include "memory_manager.h"

void GradientManager::register_gradient_map(int id, size_t size) {
    if(id <= meta_.size()) {
        meta_.resize(id + 1);
        reference_.resize(id + 1, std::make_pair(-1, nullptr));
    }
    meta_[id] = size;
}

float* GradientManager::get_gradient(int id) {
    auto& ref = reference_[id];
    if(ref.second) {
        return ref.second;
    }
    if(free_list_.empty()) {
        auto slot_id = slots_.size();
        auto sz = meta_[id];
        slots_.emplace_back(sz, 0);
        ref = std::make_pair(slot_id, slots_[slot_id].data().get());
        return ref.second;
    }
    auto slot_id = free_list_.top();
    free_list_.pop();
    auto sz = meta_[id];
    auto& vec = slots_[slot_id];
    if(sz >= vec.size()) {
        vec.resize(sz);
    }
    thrust::fill(vec.begin(), vec.end(), 0);
    ref = std::make_pair(slot_id, vec.data().get());
    return ref.second;
}

GradientDataHolder GradientManager::get_gradient_final(int id) {
    auto ptr = reference_[id].second;
    assert(ptr);
    return GradientDataHolder(*this, id, ptr);
}

GradientDataHolder::~GradientDataHolder() {
    if(id < 0) {
        return;
    }
    auto& ref = gm.reference_[id];
    gm.free_list_.push(ref.first);
    ref = std::make_pair(-1, nullptr);
}