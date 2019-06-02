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
class DeviceVector : public thrust::device_vector<T> {
  public:
    DeviceVector() = default;
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;
    using thrust::device_vector<T>::device_vector;
    operator T*() {
        return this->data().get();
    }
    operator const T*() const {
        return *this;
    }
};
