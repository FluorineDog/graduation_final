#include <math.h>
#include <cudnn.h>
#include "global.h"
#include "descriptor.h"




class CrossEntropy {
public:
    CrossEntropy(int class_size, int batch_size) : class_size(class_size), batch_size(batch_size),
                                                          dsc_io({batch_size, class_size, 1, 1}) {
    }

    void forward(float *loss, const float *act, const int *labels);

    void backward(float *act_grad, float loss_grad, const int *labels);

    size_t workspace() {
        return class_size * batch_size * sizeof(float);
    }

private:
    int class_size;
    int batch_size;
    TensorDescriptor dsc_io;
};
