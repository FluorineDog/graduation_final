#pragma once
#include "computational_graph.h"
#include "helper/defs.h"

class MetaVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override {
        weight_sz = n.functor.size_parameters();
        map_dim = n.functor.out_dim();
        workspace = 0;
    }
    virtual void visit(ActivationNode& n) override {
        weight_sz = 0;
        map_dim = n.functor.out_dim();
        workspace = 0;
    }
    virtual void visit(PlaceHolderNode& n) override {
        weight_sz = 0;
        map_dim = n.dim;
        workspace = 0;
    }
    virtual void visit(AddNode& n) override {
        weight_sz = 0;
        map_dim = n.dim;
        workspace = 0;
    }
    virtual void visit(BatchNormNode& n) override {
        weight_sz = n.functor.weight_size();
        map_dim = n.functor.out_dim();
        workspace = 0;
    }
    virtual void visit(PoolingNode& n) override {
        weight_sz = 0;
        map_dim = n.functor.dims_out();
        workspace = 0;
    }

    virtual void visit(ConvolutionNode& n) override {
        weight_sz = n.functor.get_weight_size();
        map_dim = n.functor.dims_out();
        workspace = n.functor.get_workspace_size();
    }

    struct Meta {
        size_t weight_sz;
        dim_t map_dim;
        size_t workspace;
    };

    Meta analyse(NodeBase& node) {
        node.accept(*this);
        return Meta{weight_sz, map_dim, workspace};
    }

    dim_t out_dim(NodeBase& node) {
        node.accept(*this);
        return map_dim;
    }

    size_t weight_size(NodeBase& node) {
        node.accept(*this);
        return weight_sz;
    }
    size_t workspace_size(NodeBase& node) {
        node.accept(*this);
        return workspace;
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
        auto& opt = eng.get_opt();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto weight = opt.get_weight(n.out_id);
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
    virtual void visit(BatchNormNode& n) override {
        assert(false);
        auto& mm = eng.get_mm();
        auto& opt = eng.get_opt();
        auto weight = opt.get_weight(n.out_id);
        auto in = mm.get(n.in_id);
        auto out = mm.get(n.out_id);
        n.functor.forward(out, in, weight);
    }

    virtual void visit(ConvolutionNode& n) override {
        assert(false);
        auto& mm = eng.get_mm();
        auto& opt = eng.get_opt();
        auto weight = opt.get_weight(n.out_id);
        auto in = mm.get(n.in_id);
        auto out = mm.get(n.out_id);
        n.functor.forward(out, in, weight);
    }

    virtual void visit(PoolingNode& n) override {
        assert(false);
        auto& mm = eng.get_mm();
        auto& opt = eng.get_opt();
        auto in = mm.get(n.in_id);
        auto out = mm.get(n.out_id);
        n.functor.forward(out, in);
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
    virtual void visit(BatchNormNode& n) override {}
    virtual void visit(ConvolutionNode& n) override {}
    virtual void visit(PoolingNode& n) override {}
};

class BackwardVisitor : public Visitor {
  public:
    BackwardVisitor(Engine& eng) : eng(eng) {}
    virtual void visit(FCNode& n) override {
        auto& mm = eng.get_mm();
        auto& opt = eng.get_opt();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto out_grad = mm.get(~n.out_id);
        auto in_grad = mm.get(~n.in_id);

        auto weight = opt.get_weight(n.out_id);
        auto weight_grad = opt.get_weight_grad(n.out_id);
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

    virtual void visit(BatchNormNode& n) override {
        auto& mm = eng.get_mm();
        auto& opt = eng.get_opt();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto out_grad = mm.get(~n.out_id);
        auto in_grad = mm.get(~n.in_id);
        auto weight = opt.get_weight(n.out_id);
        auto weight_grad = opt.get_weight_grad(n.out_id);
        n.functor.backward(in_grad, weight_grad, in, out_grad, weight);
 
    }

    virtual void visit(ConvolutionNode& n) override {
        auto& mm = eng.get_mm();
        auto& opt = eng.get_opt();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto out_grad = mm.get(~n.out_id);
        auto in_grad = mm.get(~n.in_id);
        auto weight = opt.get_weight(n.out_id);
        auto weight_grad = opt.get_weight_grad(n.out_id);
        n.functor.backward(in_grad, weight_grad, in, out_grad, weight);
    }

    virtual void visit(PoolingNode& n) override {
        auto& mm = eng.get_mm();
        auto& opt = eng.get_opt();
        auto out = mm.get(n.out_id);
        auto in = mm.get(n.in_id);
        auto out_grad = mm.get(~n.out_id);
        auto in_grad = mm.get(~n.in_id);
        n.functor.backward(in_grad, in, out_grad, out);
    }

    Engine& eng;
};
