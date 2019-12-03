#pragma once
#include "data_provider.h"
inline int naive_net(Engine& eng, int x_, int classes) {
    auto x = x_;
    x = eng.insert_node<ConvolutionNode>(
        x, /*C_out*/ 32,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ 2, /*dilation*/ 1);
    x = eng.insert_node<BatchNormNode>(x);
    x = eng.insert_node<ActivationNode>(x);
    x = eng.insert_node<ConvolutionNode>(
        x, /*C_out*/ 32,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ 2, /*dilation*/ 1);
    x = eng.insert_node<BatchNormNode>(x);
    x = eng.insert_node<ActivationNode>(x);
    x = eng.insert_node<ConvolutionNode>(
        x, /*C_out*/ 256,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ 2, /*dilation*/ 1);
    x = eng.insert_node<BatchNormNode>(x);
    x = eng.insert_node<ActivationNode>(x);
    x = eng.insert_node<PoolingNode>(x, /*kernel*/ 2, /*padding*/ 0, /*stride*/ 2);

    // x = eng.insert_node<FCNode>(x, hidden);
    // x = eng.insert_node<ActivationNode>(x);
    // x = eng.insert_node<FCNode>(x, hidden);
    // x = eng.insert_node<ActivationNode>(x);

    x = eng.insert_node<FCNode>(x, classes);
    return x;
}

inline int create_bottleneck(Engine& eng, int x_, int in_planes, int planes, int stride) {
    auto x = x_;
    x = eng.insert_node<ConvolutionNode>(
        x, /*C_out*/ planes,
        /*kernel*/ 1, /*group*/ 1, /*padding*/ 0,
        /*stride*/ 1, /*dilation*/ 1);
    x = eng.insert_node<BatchNormNode>(x);
    x = eng.insert_node<ActivationNode>(x);

    x = eng.insert_node<ConvolutionNode>(
        x, /*C_out*/ planes,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ stride, /*dilation*/ 1);
    x = eng.insert_node<BatchNormNode>(x);
    x = eng.insert_node<ActivationNode>(x);
    x = eng.insert_node<ConvolutionNode>(
        x, /*C_out*/ 4 * planes,
        /*kernel*/ 1, /*group*/ 1, /*padding*/ 0,
        /*stride*/ 1, /*dilation*/ 1);
    auto shortcut = x_;

    if(stride != 1 || in_planes != 4 * planes) {
        shortcut = eng.insert_node<ConvolutionNode>(
            shortcut, /*C_out*/ 4 * planes,
            /*kernel*/ 1, /*group*/ 1, /*padding*/ 0,
            /*stride*/ stride, /*dilation*/ 1);
        shortcut = eng.insert_node<BatchNormNode>(shortcut);
    }
    x = eng.insert_blend<AddNode>(x, shortcut);
    x = eng.insert_node<ActivationNode>(x);
    return x;
}

inline std::pair<int, int> make_layer(
    Engine& eng, int x_, int in_planes_,    //
    int planes, int n_blocks, int stride_) {
    auto x = x_;
    auto in_planes = in_planes_;
    auto stride = stride_;
    for(auto i : Range(n_blocks)) {
        x = create_bottleneck(eng, x, in_planes, planes, stride);
        in_planes = 4 * planes;
        stride = 1;
    }
    return std::make_pair(x, in_planes);
}

inline int construct_resnet(Engine& eng, int x_, std::vector<int> blocks, int classes) {
    assert(blocks.size() == 4);
    auto x = x_;
    auto init_planes = 64;
    x = eng.insert_node<ConvolutionNode>(
        x, /*C_out*/ init_planes,
        /*kernel*/ 3, /*group*/ 1, /*padding*/ 1,
        /*stride*/ 1, /*dilation*/ 1);
    x = eng.insert_node<BatchNormNode>(x);
    x = eng.insert_node<ActivationNode>(x);
    int in_planes = 64;
    std::tie(x, in_planes) = make_layer(eng, x, in_planes, 64, blocks[0], 1);
    std::tie(x, in_planes) = make_layer(eng, x, in_planes, 128, blocks[1], 2);
    std::tie(x, in_planes) = make_layer(eng, x, in_planes, 256, blocks[2], 2);
    std::tie(x, in_planes) = make_layer(eng, x, in_planes, 512, blocks[3], 2);
    x = eng.insert_node<PoolingNode>(x, /*kernel*/ 3, /*padding*/ 1, /*stride*/ 1);
    x = eng.insert_node<FCNode>(x, classes);
    return x;
}

inline int resnet50(Engine& eng, int x_, int classes) {
    return construct_resnet(eng, x_, {3, 4, 6, 3}, classes);
}

inline int resnet101(Engine& eng, int x_, int classes) {
    return construct_resnet(eng, x_, {3, 4, 23, 3}, classes);
}

inline int resnet152(Engine& eng, int x_, int classes) {
    return construct_resnet(eng, x_, {3, 8, 36, 3}, classes);
}


inline int resnet_inf(Engine& eng, int x_, int classes) {
    auto x = x_;
    x = eng.insert_node<PoolingNode>(x, /*kernel*/ 1,  /*padding*/ 0, /*stride*/ 1);
    return x;
    // return construct_resnet(eng, x_, {3, 4, 6, 3}, classes);
}

