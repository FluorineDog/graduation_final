#pragma once
#include "common.h"
#include "wrapper.h"
class Global {
  public:
    Global() {
        cudnnCreate(&handle_);
    }
    ~Global() {
        cudnnDestroy(handle_);
    }
    cudnnHandle_t get_handle(){
        return handle_;
    }
    void* get_workspace() {
        return workspace_.data().get();
    }
    void update_workspace_size(size_t size) {
        if(size > workspace_.size()) {
            workspace_.resize(size + size / 2);
        }
    }
    size_t get_workspace_size() {
        return workspace_.size();
    }

  private:
    device_vector<char> workspace_;
    cudnnHandle_t handle_;
};
extern Global global;