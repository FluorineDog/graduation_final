#pragma once
#include "helper/common.h"
#include "helper/defs.h"

enum class ExecType {
    forward,
    backward,
    free_feature    // and
};

struct ExecPlan {
    ExecType type;
    int node_id;
    ExecPlan(ExecType type, int node_id) : type(type), node_id(node_id) {}
};

class GradientDataHolder {
  public:
    GradientDataHolder(class GradientManager& gm, int node_id, const float* target)
        : gm(gm), node_id(node_id), target(target) {}
    GradientDataHolder(const GradientDataHolder&) = delete;
    GradientDataHolder(GradientDataHolder&& that)
        : gm(that.gm), node_id(that.node_id), target(that.target) {
        that.node_id = -1;
        that.target = nullptr;
    }
    operator const float*() {
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

class SmartManager {
  public:
    void register_node(int node_id, size_t sz);
    float* try_get_node(int node_id);
    float* prepare_new_node(int node_id);

    void free_node(int node_id);
    size_t get_node_sz(int node_id) {
        return meta_[node_id];
    }

  private:
    std::vector<std::unique_ptr<DeviceVector<float>>> slots_;
    std::vector<size_t> meta_;
    std::vector<std::tuple<int, size_t, float*>> reference_;
    std::stack<std::tuple<int, size_t, float*>> free_list_;
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
    SmartManager sm_;
};

class FeatureManager {
  public:
    FeatureManager() = default;
    std::pair<std::vector<ExecPlan>, std::vector<ExecPlan>> analyse(
        const DynamicGraph& forward_graph, int src_node, int dest_node);

    void register_feature_map(int node_id, size_t size) {
        sm_.register_node(node_id, size);
    }

    float* get_feature(int node_id) {
        if(auto ptr = sm_.try_get_node(node_id)) {
            return ptr;
        } else {
            return sm_.prepare_new_node(node_id);
        }
    }

    void terminate() {}

  private:
    SmartManager sm_;
};

class MemoryManager : public FeatureManager, public GradientManager {
  public:
    MemoryManager() = default;
    void terminate() {
        FeatureManager::terminate();
        GradientManager::terminate();
    }
};
