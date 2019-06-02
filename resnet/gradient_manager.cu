#include "memory_manager.h"
#include <tuple>

void GradientManager::register_gradient_map(int node_id, size_t size) {
    assert(node_id >= 0);
    if(node_id <= meta_.size()) {
        meta_.resize(node_id + 1, 0);
        reference_.resize(node_id + 1, std::make_tuple(-1, 0, nullptr));
    }
    assert(meta_[node_id] == 0);
    meta_[node_id] = size;
}

float* GradientManager::get_gradient(int node_id) {
    int slot_id;
    size_t slot_sz;
    float* slot_ptr;
    assert(node_id >= 0);
    {
        std::tie(slot_id, slot_sz, slot_ptr) = reference_[node_id];
        if(slot_id != -1) {
            // assert(slots_[ref.first].data().get() == ref.second);
            // assert(meta_[node_id] <= slots_[ref.first].size());
            return slot_ptr;
        }
        assert(slot_sz == 0);
        assert(slot_ptr == nullptr);
    }
    if(free_list_.empty()) {
        auto id = slots_.size();
        cout << "alloc gradient map " << id << endl;
        auto sz = meta_[node_id];
        slots_.emplace_back(std::make_unique<DeviceVector<float>>(sz, 0));
        auto ptr = slots_[id]->data().get();
        reference_[node_id] = std::make_tuple(id, sz, ptr);
        return ptr;
    }
    std::tie(slot_id, slot_sz, slot_ptr) = free_list_.top();
    free_list_.pop();
    auto std_sz = meta_[node_id];
    auto& vec = *slots_[slot_id];
    if(std_sz >= slot_sz) {
        vec.resize(std_sz);
        slot_sz = std_sz;
        slot_ptr = vec.data().get();
    }
    thrust::fill_n(vec.begin(), std_sz, 0); 
    reference_[node_id] = std::make_tuple(slot_id, slot_sz, slot_ptr);
    return slot_ptr;
}

GradientDataHolder GradientManager::get_gradient_final(int node_id) {
    assert(node_id >= 0);
    int slot_id;
    size_t slot_sz;
    float* slot_ptr;
    std::tie(slot_id, slot_sz, slot_ptr) = reference_[node_id];
    assert(slot_ptr);
    assert(slot_ptr == slots_[slot_id]->data().get());
    assert(slot_sz >= meta_[node_id]);
    return GradientDataHolder(*this, node_id, slot_ptr);
}

GradientDataHolder::~GradientDataHolder() {
    if(node_id < 0) {
        return;
    }
    auto& ref = gm.reference_[node_id];
    gm.free_list_.push(ref);
    ref = std::make_tuple(-1, 0, nullptr);
}