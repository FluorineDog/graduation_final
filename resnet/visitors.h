#pragma once

#include "helper/defs.h"
class Engine;

class MetaVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override;
    virtual void visit(ActivationNode& n) override;
    virtual void visit(PlaceHolderNode& n) override;
    virtual void visit(AddNode& n) override;
    virtual void visit(BatchNormNode& n) override;
    virtual void visit(PoolingNode& n) override;
    virtual void visit(ConvolutionNode& n) override;

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
    virtual void visit(FCNode& n) override;
    virtual void visit(ActivationNode& n) override;
    virtual void visit(PlaceHolderNode& n) override;
    virtual void visit(AddNode& n) override;
    virtual void visit(BatchNormNode& n) override;
    virtual void visit(ConvolutionNode& n) override;
    virtual void visit(PoolingNode& n) override;
    void set(float* input) {
        input_ = input;
    }

  private:
    float* input_;
    Engine& eng;
};

class FakeVisitor : public Visitor {
  public:
    virtual void visit(FCNode& n) override;
    virtual void visit(ActivationNode& n) override;
    virtual void visit(PlaceHolderNode& n) override;
    virtual void visit(AddNode& n) override;
    virtual void visit(BatchNormNode& n) override;
    virtual void visit(ConvolutionNode& n) override;
    virtual void visit(PoolingNode& n) override;
};

class BackwardVisitor : public Visitor {
  public:
    BackwardVisitor(Engine& eng) : eng(eng) {}
    virtual void visit(FCNode& n) override;
    virtual void visit(ActivationNode& n) override;
    virtual void visit(PlaceHolderNode& n) override;
    virtual void visit(AddNode& n) override;
    virtual void visit(BatchNormNode& n) override;
    virtual void visit(ConvolutionNode& n) override;
    virtual void visit(PoolingNode& n) override;
    Engine& eng;
};
