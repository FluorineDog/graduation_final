#pragma once
#include "computational_graph.h"
#include "defs.h"

class ForwardVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override {
        auto& mm = eng.get_mm();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto weight = mm.get_weight(n.out_id);
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
        assert(n.node_id == 0);
        auto dev_p = eng.get_mm().get(n.node_id);
        cudaMemcpy(dev_p, input_indicator, n.size, cudaMemcpyDefault);
    }

    virtual void visit(AddNode& n) override {
        // n
        auto& mm = eng.get_mm();
        auto out = mm.get(n.out_id);
        auto a = mm.get(n.a_id);
        auto b = mm.get(n.b_id);
        thrust::transform(a, a + n.size, b, out, thrust::plus<float>());
    }
    float* input_indicator;
    Engine& eng;
};

class FakeVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override {}
    virtual void visit(ActivationNode& n) override {}
    virtual void visit(PlaceHolderNode& n) override {}
    virtual void visit(AddNode& n) override {}
};

class MetaVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override {
        weight_sz = n.functor.size_parameters();
        map_dim = n.functor.out_dim();
    }
    virtual void visit(ActivationNode& n) override {
        weight_sz = 0;
        map_dim = n.functor.out_dim();
    }
    virtual void visit(PlaceHolderNode& n) override {
        weight_sz = 0;
        map_dim = n.dim;
    }
    virtual void visit(AddNode& n) override {
        weight_sz = 0;
        map_dim = n.dim;
    }
    struct Meta{
        size_t weight_sz;
        dim_t map_dim; 
        size_t size(){
            return get_volume(map_dim);
        }
    };

    Meta analyse(NodeBase& node) {
        node.accept(*this);
        return Meta{weight_sz, map_dim};
    }

    size_t weight_sz;
    dim_t map_dim;
};
