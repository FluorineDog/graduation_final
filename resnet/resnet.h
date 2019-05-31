#pragma once
#include "stdafx.h"
auto ref(Engine& eng, int input_dim, int B, int classes) {
    auto x = eng.insert_leaf<PlaceHolderNode>(input_dim);
    eng.src_node = x;
    auto hw = 28;
    auto c = 1;
    x = eng.insert_node<ConvolutionNode>(
        x, dim_t{B, c, hw, hw}, /*C_out*/ 32,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ 2, /*dilation*/ 1);
    hw = 14;
    c = 32;
    x = eng.insert_node<BatchNormNode>(x, dim_t{B, c, hw, hw});
    x = eng.insert_node<ActivationNode>(x, dim_t{B, c * hw * hw});
    x = eng.insert_node<ConvolutionNode>(
        x, dim_t{B, c, hw, hw}, /*C_out*/ 32,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ 2, /*dilation*/ 1);
    c = 32;
    hw = 7;
    x = eng.insert_node<BatchNormNode>(x, dim_t{B, c, hw, hw});
    x = eng.insert_node<ActivationNode>(x, dim_t{B, c * hw * hw});
    x = eng.insert_node<ConvolutionNode>(
        x, dim_t{B, c, hw, hw}, /*C_out*/ 256,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ 2, /*dilation*/ 1);
    c = 256;
    hw = 4;
    x = eng.insert_node<BatchNormNode>(x, dim_t{B, c, hw, hw});
    x = eng.insert_node<ActivationNode>(x, dim_t{B, c * hw * hw});
    x = eng.insert_node<PoolingNode>(
        x, dim_t{B, c, hw, hw}, /*kernel*/ 2, /*padding*/ 0,
        /*stride*/ 2);
    c = 256;
    hw = 2;
    x = eng.insert_node<FCNode>(x, B, c * hw * hw, classes);
    eng.dest_node = x;
    eng.finish_off();
}

int create_bottleneck(
    Engine& eng, int x_, int B, int hw_, int in_planes, int planes, int stride) {
    int c_in, hw, c_out;

    c_in = in_planes;
    c_out = planes;
    hw = hw_;
    auto x = x_;
    x = eng.insert_node<ConvolutionNode>(
        x, dim_t{B, c_in, hw, hw}, /*C_out*/ c_out,
        /*kernel*/ 1, /*group*/ 1, /*padding*/ 0,
        /*stride*/ 1, /*dilation*/ 1);
    c_in = c_out;
    x = eng.insert_node<BatchNormNode>(x, dim_t{B, c_out, hw, hw});
    x = eng.insert_node<ActivationNode>(x, dim_t{B, c_out, hw, hw});

    x = eng.insert_node<ConvolutionNode>(
        x, dim_t{B, c_in, hw, hw}, /*C_out*/ c_out,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ stride, /*dilation*/ 1);
    hw = (hw + 2 * 1 - 3) / stride + 1;
    x = eng.insert_node<BatchNormNode>(x, dim_t{B, c_out, hw, hw});
    x = eng.insert_node<ActivationNode>(x, dim_t{B, c_out, hw, hw});
    c_out = 4 * c_in;
    x = eng.insert_node<ConvolutionNode>(
        x, dim_t{B, c_in, hw, hw}, /*C_out*/ c_out,
        /*kernel*/ 1, /*group*/ 1, /*padding*/ 0,
        /*stride*/ 1, /*dilation*/ 1);
    c_in = c_out;
    auto shortcut = x_;
    if(stride != 1 || in_planes != 4 * planes) {
        auto c_in = in_planes;
        auto c_out = 4 * planes;
        auto hw = hw_;
        shortcut = eng.insert_node<ConvolutionNode>(
            shortcut, dim_t{B, c_in, hw, hw}, /*C_out*/ c_out,
            /*kernel*/ 1, /*group*/ 1, /*padding*/ 0,
            /*stride*/ stride, /*dilation*/ 1);
        c_in = c_out;
        hw = (hw - 1) / stride + 1;
        shortcut = eng.insert_node<BatchNormNode>(shortcut, dim_t{B, c_out, hw, hw});
    }
    x = eng.insert_blend<AddNode>(x, shortcut, dim_t{B, c_in, hw, hw});
    x = eng.insert_node<ActivationNode>(x, dim_t{B, c_in, hw, hw});
    return x;
}

void make_layer(Engine& eng, int x, int in_planes, int planes, int n_blocks, int stride_){
    std::vector<int> strides(n_blocks, 1);
    strides[0] = stride_;
    for(auto stride: strides){
            
    }
    
}

void construct_resnet(Engine& eng) {
    
}