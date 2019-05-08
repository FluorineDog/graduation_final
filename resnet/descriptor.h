#include "common.h"
class TensorDescriptor {
  public:
    TensorDescriptor(){
        cudnnCreateTensorDescriptor(&desc_);
    }
    operator cudnnTensorDescriptor_t(){
        return desc_; 
    }

    
    ~TensorDescriptor(){
        cudnnDestroyTensorDescriptor(desc_);
    }
  private:
    cudnnTensorDescriptor_t desc_;
};
