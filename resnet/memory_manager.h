#pragma once
#include "helper/common.h"
#include "helper/defs.h"
class MemoryManager {
  public:
    void init() {
        //
    }
    void register_feature_map(int id, size_t size) {
        assert(!feature_mapping.count(id));
        feature_mapping[id].resize(size);
    }
    void register_gradient_map(int id, size_t size) {
        assert(!gradient_mapping.count(id));
        gradient_mapping[id].resize(size);
    }
    float* get_feature(int id) {
        assert(feature_mapping.count(id));
        return feature_mapping[id];
    }

    float* get_gradient(int id) {
        assert(gradient_mapping.count(id));
        return gradient_mapping[id];
    }

    const float* get_gradient_final(int id) {
        assert(gradient_mapping.count(id));
        return gradient_mapping[id];
    }

    void zero_grad() {
        #pragma omp parallel for
        for(auto& pr : gradient_mapping) {
            // if(pr.first > 0){
            //     break;
            // }
            auto& vec = pr.second;
            auto ptr = vec.data().get();
            thrust::fill(thrust::device, vec.begin(), vec.end(), 0);
            // cudaMemset(vec.data().get(), 0, sizeof(float) * vec.size());
            // assert(inspect(ptr) == 0);
        }
    }

    void free(int id) {}
    void terminate() {
        feature_mapping.clear();
        gradient_mapping.clear();
    }

  private:
    std::map<int, DeviceVector<float>> feature_mapping;
    std::map<int, DeviceVector<float>> gradient_mapping;
};