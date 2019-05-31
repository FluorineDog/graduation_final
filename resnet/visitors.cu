#include "visitors.h"
#include "computational_graph.h"
void MetaVisitor::visit(FCNode& n) {
    weight_sz = n.functor.size_parameters();
    map_dim = n.functor.out_dim();
    workspace = 0;
}
void MetaVisitor::visit(ActivationNode& n) {
    weight_sz = 0;
    map_dim = n.functor.out_dim();
    workspace = 0;
}
void MetaVisitor::visit(PlaceHolderNode& n) {
    weight_sz = 0;
    map_dim = n.dim;
    workspace = 0;
}
void MetaVisitor::visit(AddNode& n) {
    weight_sz = 0;
    map_dim = n.dim;
    workspace = 0;
}
void MetaVisitor::visit(BatchNormNode& n) {
    weight_sz = n.functor.weight_size();
    map_dim = n.functor.out_dim();
    workspace = 0;
}
void MetaVisitor::visit(PoolingNode& n) {
    weight_sz = 0;
    map_dim = n.functor.dims_out();
    workspace = 0;
}

void MetaVisitor::visit(ConvolutionNode& n) {
    weight_sz = n.functor.get_weight_size();
    map_dim = n.functor.dims_out();
    workspace = n.functor.get_workspace_size();
}

void ForwardVisitor::visit(FCNode& n) {
    auto& mm = eng.get_mm();
    auto& opt = eng.get_opt();
    auto out = mm.get(n.out_id);
    auto in = mm.get(n.in_id);
    auto weight = opt.get_weight(n.out_id);
    n.functor.forward(out, in, weight);
}
void ForwardVisitor::visit(ActivationNode& n) {
    auto& mm = eng.get_mm();
    auto out = mm.get(n.out_id);
    auto in = mm.get(n.in_id);
    n.functor.forward(out, in);
}

void ForwardVisitor::visit(PlaceHolderNode& n) {
    //
    assert(n.node_id == 0);
    auto dev_p = eng.get_mm().get(n.node_id);
    cudaMemcpy(dev_p, input_, n.size * sizeof(T), cudaMemcpyDefault);
}

void ForwardVisitor::visit(AddNode& n) {
    // n
    auto& mm = eng.get_mm();
    auto out = mm.get(n.out_id);
    auto a = mm.get(n.a_id);
    auto b = mm.get(n.b_id);
    thrust::transform(thrust::device, a, a + n.size, b, out, thrust::plus<float>());
}

void ForwardVisitor::visit(BatchNormNode& n) {
    // assert(false);
    auto& mm = eng.get_mm();
    auto& opt = eng.get_opt();
    auto weight = opt.get_weight(n.out_id);
    auto in = mm.get(n.in_id);
    auto out = mm.get(n.out_id);
    n.functor.forward(out, in, weight);
}

void ForwardVisitor::visit(ConvolutionNode& n) {
    // assert(false);
    auto& mm = eng.get_mm();
    auto& opt = eng.get_opt();
    auto weight = opt.get_weight(n.out_id);
    auto in = mm.get(n.in_id);
    auto out = mm.get(n.out_id);
    n.functor.forward(out, in, weight);
}

void ForwardVisitor::visit(PoolingNode& n) {
    // assert(false);
    auto& mm = eng.get_mm();
    auto& opt = eng.get_opt();
    auto in = mm.get(n.in_id);
    auto out = mm.get(n.out_id);
    n.functor.forward(out, in);
}

void FakeVisitor::visit(FCNode& n) {}
void FakeVisitor::visit(ActivationNode& n) {}
void FakeVisitor::visit(PlaceHolderNode& n) {}
void FakeVisitor::visit(AddNode& n) {}
void FakeVisitor::visit(BatchNormNode& n) {}
void FakeVisitor::visit(ConvolutionNode& n) {}
void FakeVisitor::visit(PoolingNode& n) {}

void BackwardVisitor::visit(FCNode& n) {
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

void BackwardVisitor::visit(ActivationNode& n) {
    auto& mm = eng.get_mm();
    auto out = mm.get(n.out_id);
    auto in = mm.get(n.in_id);
    auto out_grad = mm.get(~n.out_id);
    auto in_grad = mm.get(~n.in_id);
    n.functor.backward(in_grad, out_grad, in, out);
}

void BackwardVisitor::visit(PlaceHolderNode& n) {
    return;
}
void BackwardVisitor::visit(AddNode& n) {
    auto& mm = eng.get_mm();
    auto a_g = mm.get(~n.a_id);
    auto b_g = mm.get(~n.b_id);
    auto out_grad = mm.get(~n.out_id);
    thrust::transform(
        thrust::device, a_g, a_g + n.size, out_grad, a_g, thrust::plus<double>());
    thrust::transform(
        thrust::device, b_g, b_g + n.size, out_grad, b_g, thrust::plus<double>());
}

void BackwardVisitor::visit(BatchNormNode& n) {
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

void BackwardVisitor::visit(ConvolutionNode& n) {
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

void BackwardVisitor::visit(PoolingNode& n) {
    auto& mm = eng.get_mm();
    auto& opt = eng.get_opt();
    auto out = mm.get(n.out_id);
    auto in = mm.get(n.in_id);
    auto out_grad = mm.get(~n.out_id);
    auto in_grad = mm.get(~n.in_id);
    n.functor.backward(in_grad, in, out_grad, out);
}
