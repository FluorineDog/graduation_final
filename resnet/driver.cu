#include "../doglib/graph/procedure.h"
#include "cg.h"
#include "cross_entropy.h"
#include <random>

Global global;
int main() {
    Engine eng;
    // define network structure
    int B = 128;
    int features = 1000;
    int hidden = 512;
    int classes = 2;
    dim_t input_dim = {B, features};

    auto x = eng.insert_leaf<PlaceHolderNode>(input_dim);
    eng.src_node = x;
    auto shortcut = x;
    x = eng.insert_node<FCNode>(x, B, features, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_node<FCNode>(x, B, hidden, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_blend<AddNode>(x, shortcut, dim_t{B, hidden});
    x = eng.insert_node<FCNode>(x, B, hidden, classes);
    eng.dest_node = x;
    eng.finish_off();

    host_vector<float> input;
    input.resize(B * 1000);
    std::default_random_engine e;
    for(auto& x : input) {
        x = 1.0 * (float)(e() % 10001) / 10000;
    }

    host_vector<int> labels;
    for(auto id: Range(B)){
        float sum;
        for(auto x: Range(features)){
            sum += input[id * features + x]; 
        }
        int label = sum > 1000 ? 1:0;
        labels.push_back(label);
    }
    device_vector<int> dev_labels = labels;
    DeviceVector<T> losses(B);
    CrossEntropy ce(B, classes); 

    eng.forward_pass(input.data());
    auto act = eng.get_ptr(eng.dest_node);
    auto act_grad = eng.get_ptr(~eng.dest_node);
    ce.forward(losses, act, dev_labels.data().get());
    // ce.backward(act_grad, losses, dev_labels.data().get());
    
 }