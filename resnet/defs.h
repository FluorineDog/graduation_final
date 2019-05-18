#pragma once
#include "descriptor.h"
#include "global.h"
#include "fc.h"
#include "bn.h"
#include "activation.h"

struct NodeBase {
    virtual void visit(class Visitor&) {}
    virtual ~NodeBase() = default;
};

class Visitor {
  public:
    virtual void visit(class FCNode&) = 0;
    virtual void visit(class ActivationNode&) = 0;
    virtual void visit(class PlaceHolderNode&) = 0;
    virtual void visit(class VariableNode&) = 0;
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
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
};

struct ActivationNode : NodeBase {
    ActivationNode(int in_id, int out_id, dim_t dim)
        : in_id(in_id), out_id(out_id), functor(dim) {}
    int in_id;
    int out_id;
    ActivationFunctor functor;
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
};

struct PlaceHolderNode : NodeBase {
    PlaceHolderNode(int x, dim_t dim) : node_id(x), dim(dim) {}
    int node_id;
    dim_t dim;
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
};

struct VariableNode : NodeBase {
    VariableNode(int x) : node_id(x) {}
    int node_id;
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
};