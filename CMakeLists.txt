cmake_minimum_required(VERSION 3.10)
project(cudnn_resnet LANGUAGES CXX CUDA)
find_package(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 --gpu-architecture=compute_61 --gpu-code=sm_61 -std=c++14)
add_subdirectory(conv_sample)
add_subdirectory(resnet)

