#include "memory_manager.h"
#include <tuple>

void GradientManager::register_gradient_map(int node_id, size_t size) {
    assert(node_id >= 0);
    sm_.register_node(node_id, size);
}

void SmartManager::register_node(int node_id, size_t size) {
    if(node_id <= meta_.size()) {
        meta_.resize(node_id + 1, 0);
        reference_.resize(node_id + 1, std::make_tuple(-1, 0, nullptr));
    }
    assert(meta_[node_id] == 0);
    meta_[node_id] = size;
}

float* GradientManager::get_gradient(int node_id) {
    assert(node_id >= 0);
    if(auto ptr = sm_.try_get_node(node_id)) {
        return ptr;
    }
    auto ptr = sm_.prepare_new_node(node_id);
    auto sz = sm_.get_node_sz(node_id);
    thrust::fill_n(thrust::device, ptr, sz, 0);
    return ptr;
}

float* SmartManager::try_get_node(int node_id) {
    int slot_id;
    size_t slot_sz;
    float* slot_ptr;
    std::tie(slot_id, slot_sz, slot_ptr) = reference_[node_id];
    if(slot_id != -1) {
        return slot_ptr;
    } else {
        assert(slot_id == -1);
        assert(slot_sz == 0);
        assert(slot_ptr == 0);
        return nullptr;
    }
}

float* SmartManager::prepare_new_node(int node_id) {
    assert(try_get_node(node_id) == nullptr);
    int slot_id;
    size_t slot_sz;
    float* slot_ptr;
    if(free_list_.empty()) {
        auto id = slots_.size();
        cout << "[alloc map " << id << "] ";
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
    // thrust::fill_n(vec.begin(), std_sz, 0);
    reference_[node_id] = std::make_tuple(slot_id, slot_sz, slot_ptr);
    return slot_ptr;
}

GradientDataHolder GradientManager::get_gradient_final(int node_id) {
    assert(node_id >= 0);
    auto slot_ptr = sm_.try_get_node(node_id);
    assert(slot_ptr);
    return GradientDataHolder(*this, node_id, slot_ptr);
}
void SmartManager::free_node(int node_id) {
    auto& ref = reference_[node_id];
    free_list_.push(ref);
    ref = std::make_tuple(-1, 0, nullptr);
}

GradientDataHolder::~GradientDataHolder() {
    if(node_id < 0) {
        return;
    }
    gm.sm_.free_node(node_id);
}