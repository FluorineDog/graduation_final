#pragma once

#include "../helper/common.h"
#include <random>
#include <thrust/count.h>
#include <iomanip>
#include <thrust/device_vector.h>

inline void dog_print(std::string name, const float* ptr, const dim_t& dim) {
    cout << name << endl;
    auto sz = get_volume(dim);

    cudaDeviceSynchronize();
    host_vector<T> vec(sz);
    cudaMemcpy(vec.data(), ptr, sz * sizeof(float), cudaMemcpyDefault);
    auto tmp = dim;
    std::reverse(tmp.begin(), tmp.end());
    for(auto index : Range(sz)) {
        int index_cpy = index;
        for(auto x : tmp) {
            if(index_cpy % x != 0) break;
            index_cpy /= x;
            cout << "--------" << endl;
        }
        cout.precision(3);
        cout << std::setw(6) << vec[index] << " ";
    }
    cout << endl << "##########" << endl;
}
inline float inspect(float* dev_ptr) {
    float tmp;
    cudaMemcpy(&tmp, dev_ptr, sizeof(float), cudaMemcpyDefault);
    return tmp;
}

inline void check(cudnnStatus_t status){
    assert(status == CUDNN_STATUS_SUCCESS);
}