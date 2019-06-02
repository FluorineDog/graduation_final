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

float get_acc(float* dev_logits, int* labels, int N, int feature);
host_vector<int> get_labels();
host_vector<float> get_data();
