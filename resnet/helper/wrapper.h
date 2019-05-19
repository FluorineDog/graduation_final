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
    using thrust::device_vector<T>::device_vector;
    operator T*() {
        return this->data().get();
    }
    operator T*() const {
        return *this;
    }
};
