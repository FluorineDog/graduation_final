#pragma once
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../../doglib/common/common.h"
#include "../../doglib/graph/procedure.h"
#include <cuda_runtime.h>
using namespace doglib::common;
using doglib::graph::DynamicGraph;
using doglib::graph::ProcedureDFS;
using ull = long long;
using T = float;
using namespace thrust;
using std::cout;
using std::endl;
using std::vector;
