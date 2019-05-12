#include "functor.h"
#include "cublas.h"



class FCFunctor {
  public:
    FCFunctor(int batch, int in_size, int out_size)
        :batch(batch), in_size(in_size), out_size(out_size) {
    }
    void forward(void* out, const void* in, const void* weight) {
        auto bias = (float* )weight +  in_size * out_size;
        cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, batch, in_size, out_size, 1.0, (float*)in, in_size, (float*)weight, )
    }
    void backwardFilter(void* weight_grad, const void* in, const void* out_grad) {}
    void backwardData(void* in_grad, const void* weight, const void* out_grad) {}
    void backward(void* in_grad, void* weight_grad, const void* in, const void* out_grad,
                  const void* weight) {
        backwardFilter(weight_grad, in, out_grad);
        backwardData(in_grad, weight, out_grad);
    }

  private:
    int batch;
    int in_size;
    int out_size;
};