cmake_minimum_required(VERSION 3.10)
project(cudnn_resnet LANGUAGES CXX CUDA)
find_package(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 --expt-extended-lambda)
#add_subdirectory(conv_sample)
add_subdirectory(resnet)

