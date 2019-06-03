#pragma once
#include "common.h"
#include "wrapper.h"
#include "cublas.h"
class Global {
  public:
    Global() {
        auto st = cudnnCreate(&cudnn_handle_);
        assert(st == CUDNN_STATUS_SUCCESS);
        cublasCreate_v2(&blas_handle_);
    }
    ~Global() {
        cudnnDestroy(cudnn_handle_);
        cublasDestroy_v2(blas_handle_);
    }

    cudnnHandle_t cudnn_handle() {
        return cudnn_handle_;
    }

    cublasHandle_t cublas_handle() {
        return blas_handle_;
    }

    float* get_workspace() {
        return workspace_.data().get();
    }
    void update_workspace_size(size_t size) {
        size = (size + sizeof(float)) / sizeof(float);
        if(size > workspace_.size()) {
            workspace_.resize(size + size / 2);
        }
    }
    size_t get_workspace_size() {
        return workspace_.size() * sizeof(float);
    }
    bool is_training(){
        return training_;
    }
    void set_training(bool training){
        this->training_ = training; 
    }
    size_t round = 0;
  private:
    bool training_ = true;
    device_vector<float> workspace_;
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t blas_handle_;
};
extern Global global;