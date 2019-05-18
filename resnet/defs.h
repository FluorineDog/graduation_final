#pragma once
#include "descriptor.h"
#include "global.h"
#include "fc.h"
#include "bn.h"
#include "activation.h"

struct NodeBase {
    virtual void visit(class Visitor&) = 0;
    virtual ~NodeBase() = default;
    int size;
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
    int in_id;
    int out_id;
    int weight_id;
    FCFunctor functor;
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
};

struct ActivationNode : NodeBase {
    int in_id;
    int out_id;
    ActivationFunctor functor;
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
};

struct PlaceHolderNode: NodeBase {
    PlaceHolderNode(int x):node_id(x) {}
    int node_id;
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
}l;

struct VariableNode: NodeBase {
    VariableNode(int x):node_id(x) {}
    int node_id;
    void visit(Visitor& v) override {
        return v.visit(*this);
    }
};