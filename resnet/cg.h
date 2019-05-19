#pragma once
#include "computational_graph.h"
#include "helper/defs.h"

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
    struct Meta {
        size_t weight_sz;
        dim_t map_dim;
    };

    Meta analyse(NodeBase& node) {
        node.accept(*this);
        return Meta{weight_sz, map_dim};
    }

    dim_t out_dim(NodeBase& node) {
        node.accept(*this);
        return map_dim;
    }

    size_t weight_size(NodeBase& node) {
        return weight_sz;
    }

  private:
    dim_t map_dim;
    size_t weight_sz;
    size_t workspace = 0;
};

class ForwardVisitor : public Visitor {
  public:
    ForwardVisitor(Engine& eng) : eng(eng), input_(nullptr) {}
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
        cudaMemcpy(dev_p, input_, n.size * sizeof(T), cudaMemcpyDefault);
    }

    virtual void visit(AddNode& n) override {
        // n
        auto& mm = eng.get_mm();
        auto out = mm.get(n.out_id);
        auto a = mm.get(n.a_id);
        auto b = mm.get(n.b_id);
        thrust::transform(thrust::device, a, a + n.size, b, out, thrust::plus<float>());
    }
    void set(float* input) {
        input_ = input;
    }

  private:
    float* input_;
    Engine& eng;
};

class FakeVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override {}
    virtual void visit(ActivationNode& n) override {}
    virtual void visit(PlaceHolderNode& n) override {}
    virtual void visit(AddNode& n) override {}
};

class BackwardVisitor : public Visitor {
  public:
    BackwardVisitor(Engine& eng) : eng(eng) {}
    virtual void visit(FCNode& n) override {
        auto& mm = eng.get_mm();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto out_grad = mm.get(~n.out_id);
        auto in_grad = mm.get(~n.in_id);
        auto weight = mm.get_weight(n.out_id);
        auto weight_grad = mm.get_weight_grad(n.out_id);
        n.functor.backward(in_grad, weight_grad, in, out_grad, weight);
    }
    virtual void visit(ActivationNode& n) override {
        auto& mm = eng.get_mm();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto out_grad = mm.get(~n.out_id);
        auto in_grad = mm.get(~n.in_id);
        n.functor.backward(in_grad, out_grad, in, out);
    }

    virtual void visit(PlaceHolderNode& n) override {
        return;
    }
    virtual void visit(AddNode& n) override {
        auto& mm = eng.get_mm();
        auto a_g = mm.get(~n.a_id);
        auto b_g = mm.get(~n.b_id);
        auto out_grad = mm.get(~n.out_id);
        thrust::transform(thrust::device, a_g, a_g + n.size, out_grad, a_g,
                          thrust::plus<double>());
        thrust::transform(thrust::device, b_g, b_g + n.size, out_grad, b_g,
                          thrust::plus<double>());
    }
    Engine& eng;
};
