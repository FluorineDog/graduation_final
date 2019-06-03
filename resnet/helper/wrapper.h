#pragma once
#include "common.h"
class Dims : public std::vector<int> {
  public:
    using std::vector<int>::vector;
    operator int*() {
        return this->data();
    }
    operator int*() const {
        return *this;
    }
};

template<class T>
class DeviceVector {
  public:
    DeviceVector() = default;
    DeviceVector(size_t size){
        resize(size);
    }
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;
    // using thrust::device_vector<T>::device_vector;
    void resize(size_t size) {
        assert(size > size_);
        size_ = size;
        if(ptr){
            cudaFree(ptr);
        }
        auto st = cudaMalloc(&ptr, size * sizeof(T));
        assert(!st);
    }
    float* begin() {
        return ptr;
    }
    float* end() {
        return ptr + size_;
    }
    operator T*() {
        // return this->data().get();
        return ptr;
    }
    operator const T*() const {
        return *this;
    }
    size_t size(){
        return size_;
    }
    ~DeviceVector(){
         
    }
  private:
    size_t size_ = 0; 
    T* ptr = nullptr;
};
