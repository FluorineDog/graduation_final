#pragma once
#include "helper/common.h"
#include "helper/defs.h"
class GradientDataHolder {
  public:
    GradientDataHolder(class GradientManager& gm, int node_id, const float* target) : gm(gm), node_id(node_id), target(target) {}
    GradientDataHolder(const GradientDataHolder&) = delete;
    GradientDataHolder(GradientDataHolder&& that)
        : gm(that.gm), node_id(that.node_id), target(that.target) {
        that.node_id = -1;
        that.target = nullptr;
    }
    operator const float*(){
        return target;
    }
    GradientDataHolder& operator=(const GradientDataHolder&) = delete;
    GradientDataHolder& operator=(GradientDataHolder&& that) = delete;
    ~GradientDataHolder();

  private:
    class GradientManager& gm;
    int node_id;
    const float* target;
};

class GradientManager {
  public:
  
    void register_gradient_map(int node_id, size_t size);
    float* get_gradient(int node_id);
    GradientDataHolder get_gradient_final(int node_id);

    void zero_grad() {}
    void terminate() {}
  private:
    friend GradientDataHolder;
    std::vector<std::unique_ptr<DeviceVector<float>>> slots_;
    std::vector<size_t> meta_;
    std::vector<std::tuple<int, size_t, float*>> reference_;
    std::stack<std::tuple<int, size_t, float*>> free_list_;
};

class MemoryManager : public GradientManager {
  public:
    void register_feature_map(int node_id, size_t size) {
        assert(!feature_mapping.count(node_id));
        feature_mapping[node_id].resize(size);
    }

    float* get_feature(int node_id) {
        assert(feature_mapping.count(node_id));
        return feature_mapping[node_id];
    }

    void terminate() {
        feature_mapping.clear();
        GradientManager::terminate();
    }

  private:
    std::map<int, DeviceVector<float>> feature_mapping;
};