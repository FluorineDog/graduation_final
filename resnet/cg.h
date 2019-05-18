#pragma once
#include "computational_graph.h"
#include "defs.h"

class ForwardVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override {
        auto& mm = eng.get_mm();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto weight = mm.get(n.weight_id);
        n.functor.forward(out, in, weight);
    }
    virtual void visit(ActivationNode& n) override {
        auto& mm = eng.get_mm();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        n.functor.forward(out, in);
    }
    virtual void visit(PlaceHolderNode& n) override {
        //
    }
    virtual void visit(VariableNode& n) override {
        //
    }
    Engine& eng;
};

class FakeVisitor : public Visitor {
    virtual void visit(FCNode& n) override {}
    virtual void visit(ActivationNode& n) override {}
    virtual void visit(PlaceHolderNode& n) override {}
    virtual void visit(VariableNode& n) override {}
};