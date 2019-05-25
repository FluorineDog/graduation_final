#pragma once
#include "../doglib/graph/procedure.h"
#include "visitors.h"
#include "components/cross_entropy.h"
#include <random>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <fstream>
struct functor {
    __host__ __device__ bool operator()(float x) {
        return x < 0.6931471805599453;
    }
};

// void dog_print(std::string name, const float* ptr, const dim_t& dim) {
//     cout << name << endl;
//     auto sz = get_volume(dim);

//     cudaDeviceSynchronize();
//     host_vector<T> vec(sz);
//     cudaMemcpy(vec.data(), ptr, sz * sizeof(float), cudaMemcpyDefault);
//     auto tmp = dim;
//     std::reverse(tmp.begin(), tmp.end());
//     for(auto index : Range(sz)) {
//         int index_cpy = index;
//         for(auto x : tmp) {
//             if(index_cpy % x != 0) break;
//             index_cpy /= x;
//             cout << "--------" << endl;
//         }
//         cout << vec[index] << " ";
//     }
//     cout << endl << "##########" << endl;
// }



float get_acc(float* dev_logits, int* labels, int N, int feature);
host_vector<int> get_labels();
host_vector<float> get_data();
