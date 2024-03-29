#pragma once
#include "../components/descriptor.h"
#include "global.h"
#include "../components/fc.h"
#include "../components/bn.h"
#include "../components/pooling.h"
#include "../components/conv.h"
#include "../components/activation.h"

struct NodeBase {
    virtual void accept(class Visitor&) = 0;
    virtual ~NodeBase() = default;
};

class Visitor {
  public:
    virtual void visit(class FCNode&) = 0;
    virtual void visit(class ActivationNode&) = 0;
    virtual void visit(class PlaceHolderNode&) = 0;
    // virtual void visit(class VariableNode&) = 0;
    virtual void visit(class AddNode&) = 0;
    virtual void visit(class BatchNormNode&) = 0;
    virtual void visit(class PoolingNode&) = 0;
    virtual void visit(class ConvolutionNode&) = 0;
    ~Visitor() = default;
};

// start from here
struct FCNode : NodeBase {
    FCNode(int in_id, int out_id, dim_t in, int out_size)
        : in_id(in_id),
          out_id(out_id),
          functor(in[0], get_volume(in) / in[0], out_size) {}
    int in_id;
    int out_id;
    FCFunctor functor;
    void accept(Visitor& v) override {
        return v.visit(*this);
    }
};

struct AddNode : NodeBase {
    AddNode(int a_id, int b_id, int out_id, dim_t dim)
        : a_id(a_id), b_id(b_id), out_id(out_id), dim(dim), size(get_volume(dim)) {}
    int a_id;
    int b_id;
    int out_id;
    dim_t dim;
    size_t size;
    void accept(Visitor& v) override {
        return v.visit(*this);
    }
};

struct ActivationNode : NodeBase {
    ActivationNode(int in_id, int out_id, dim_t dim)
        : in_id(in_id), out_id(out_id), functor(dim) {}
    int in_id;
    int out_id;
    ActivationFunctor functor;
    void accept(Visitor& v) override {
        return v.visit(*this);
    }
};

struct PlaceHolderNode : NodeBase {
    PlaceHolderNode(int x, dim_t dim) : node_id(x), dim(dim), size(get_volume(dim)) {}
    int node_id;
    dim_t dim;
    int size;
    void accept(Visitor& v) override {
        return v.visit(*this);
    }
};

struct BatchNormNode : NodeBase {
    BatchNormNode(int in_id, int out_id, dim_t dim)
        : in_id(in_id), out_id(out_id), functor(dim) {}
    int in_id;
    int out_id;
    BatchNorm functor;
    void accept(Visitor& v) override {
        return v.visit(*this);
    }
};

struct PoolingNode : NodeBase {
    PoolingNode(int in_id, int out_id, dim_t in, int K, int padding, int stride)
        : in_id(in_id), out_id(out_id), functor(in, K, padding, stride) {}
    int in_id;
    int out_id;
    PoolingFunctor functor;
    void accept(Visitor& v) override {
        return v.visit(*this);
    }
};

struct ConvolutionNode : NodeBase {
    ConvolutionNode(
        int in_id, int out_id, dim_t dims_in, int C_out, int K, int group, int padding,
        int stride, int dilation = 1)
        : in_id(in_id),
          out_id(out_id),
          functor(dims_in, C_out, K, group, padding, stride, dilation) {}
    int in_id;
    int out_id;
    ConvolutionFunctor functor;
    void accept(Visitor& v) override {
        return v.visit(*this);
    }
};

// todo batchnorm, pooling, conv