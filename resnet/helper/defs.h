#pragma once
#include "../descriptor.h"
#include "global.h"
#include "../components/fc.h"
#include "../components/bn.h"
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
    // virtual void visit(class BatchNormNode& ) = 0;
    ~Visitor() = default;
};

// start from here
struct FCNode : NodeBase {
    FCNode(int in_id, int out_id, int batch, int in_size, int out_size)
        : in_id(in_id), out_id(out_id), functor(batch, in_size, out_size) {}
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

// todo batchnorm, pooling, conv