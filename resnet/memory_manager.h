#pragma once
#include "helper/common.h"
#include "helper/defs.h"
class GradientDataHolder {
  public:
    GradientDataHolder(class GradientManager& gm, int id, const float* target) : gm(gm), id(id), target(target) {}
    GradientDataHolder(const GradientDataHolder&) = delete;
    GradientDataHolder(GradientDataHolder&& that)
        : gm(that.gm), id(that.id), target(that.target) {
        that.id = -1;
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
    int id;
    const float* target;
};

class GradientManager {
  public:
  
    void register_gradient_map(int id, size_t size);
    float* get_gradient(int id);
    GradientDataHolder get_gradient_final(int id);

    void zero_grad() {}
    void terminate() {}
  private:
    friend GradientDataHolder;
    std::vector<DeviceVector<float>> slots_;
    std::vector<size_t> meta_;
    std::vector<std::pair<int, float*>> reference_;
    std::stack<int> free_list_;
};

class MemoryManager : public GradientManager {
  public:
    void register_feature_map(int id, size_t size) {
        assert(!feature_mapping.count(id));
        feature_mapping[id].resize(size);
    }

    float* get_feature(int id) {
        assert(feature_mapping.count(id));
        return feature_mapping[id];
    }

    void terminate() {
        feature_mapping.clear();
        GradientManager::terminate();
    }

  private:
    std::map<int, DeviceVector<float>> feature_mapping;
};