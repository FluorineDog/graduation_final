#pragma once
#include "helper/common.h"
#include "helper/defs.h"
#include <stdlib.h>
#include <numeric>

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
constexpr size_t inf = std::numeric_limits<size_t>::max() / 10;


using SlotMeta = std::tuple<int, size_t, float*>;
struct SmartGlobal{
   std::multimap<size_t, SlotMeta> free_lists_;
   int step = 0;
   size_t last_round = 0;
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
    static SlotMeta get_best(size_t sz) {
        auto& free_lists_ = sg_.free_lists_;
        if(free_lists_.size() == 0) {
            return std::make_tuple(-1, 0, nullptr);
        }
        auto iter = free_lists_.lower_bound(sz);
        if(iter == free_lists_.end()) {
            --iter;
        }
        auto res = iter->second;
        free_lists_.erase(iter);
        return res;
    }
    static void return_free(SlotMeta x) {
        auto sz = std::get<1>(x);
        sg_.free_lists_.emplace(sz, x);
    }

    // SlotMeta get_best(size_t sz){
    //     if(fl_.empty()){
    //         return std::make_tuple(-1, 0, nullptr);
    //     }
    //     auto res = fl_.top();
    //     fl_.pop();
    //     return res;
    // }
    // void return_free(SlotMeta x) {
    //     fl_.push(x);
    // }

  private:
    static std::vector<std::unique_ptr<DeviceVector<float>>> slots_;
    static SmartGlobal sg_;
    std::vector<size_t> meta_;
    std::vector<SlotMeta> reference_;
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
        auto ptr = sm_.try_get_node(node_id);
        assert(ptr);
        return ptr;
    }

    float* get_feature_write(int node_id) {
        if(auto ptr = sm_.try_get_node(node_id)) {
            return ptr;
        } else {
            return sm_.prepare_new_node(node_id);
        }
    }

    void free_feature(int node_id) {
        assert(node_id >= 0);
        sm_.free_node(node_id);
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
