#pragma once
#include "descriptor.h"
#include "global.h"
#include "fc.h"
#include "bn.h"
#include "activation.h"

struct NodeBase {
    virtual void visit(class Visitor&) = 0;
    virtual ~NodeBase() = default;
};

class Visitor {
  public:
    virtual void visit(class FCNode&) = 0;
    virtual void visit(class ActivationNode&) = 0;
    // virtual void visit(class BatchNormNode& ) = 0;
    ~Visitor() = default;
};

struct FCNode : NodeBase {
    int in_id;
    int out_id;
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
