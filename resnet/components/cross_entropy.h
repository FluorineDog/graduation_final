#pragma once
#include <math.h>
#include <cudnn.h>
#include "../helper/global.h"
#include "../descriptor.h"

class CrossEntropy {
  public:
    CrossEntropy(int batch_size, int class_size)
        : batch_size(batch_size),
          class_size(class_size),
          dsc_io({batch_size, class_size, 1, 1}) {}

    void forward(float *loss, const float *act, const int *labels);

    void backward(float *act_grad, float rate, const float* act, const float* loss_grad, const int *labels);

    size_t workspace() {
        return class_size * batch_size * sizeof(float) + 16;
    }

  private:
    int batch_size;
    int class_size;
    TensorDescriptor dsc_io;
};
