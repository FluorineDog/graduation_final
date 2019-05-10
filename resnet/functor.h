#pragma once
#include "descriptor.h"
class Workspace {
  public:
};
class ConvolutionFunctor {
  public:
    void forward(void* ptr_out_, const void* ptr_in_, const void* ptr_filter_,
                 void* workspace) {
        //
        float alpha = 1, beta = 0;
        // cudnnConvolutionForward(handle, &alpha,                   //
        //                         dsc_in, ptr_in_,                   //
        //                         dsc_filter, ptr_filter_,           //
        //                         dsc_conv, kAlgo,                  //
        //                         workspace, workspace_size,    //
        //                         &beta,                            //
        //                         dsc_out, vec_out                  //
        // );
    }
    void backwardData(void* ptr_out_, const void* ptr_in_, const void* ptr_filter_,
                      void* workspace) {}
    void backwardFilter(void* ptr_out_, const void* ptr_in_, const void* ptr_filter_,
                        void* workspace) {}
    

  private:
    ConvolutionDescriptor dsc_conv;
    TensorDescriptor dsc_in;
    TensorDescriptor dsc_out;
    FilterDescriptor dsc_filter;
};
