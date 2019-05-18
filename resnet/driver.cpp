#include "../doglib/graph/procedure.h"
#include "cg.h"
#include <random>

int main() {
    Engine eng;
    // define network structure
    int B = 128;
    int features = 1000;
    int hidden = 512;
    dim_t input_dim = {B, features};

    auto x = eng.insert_leaf<PlaceHolderNode>(input_dim);
    eng.src_node = x;
    auto shortcut = x;
    x = eng.insert_node<FCNode>(x, B, features, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_node<FCNode>(x, B, hidden, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_node<AddNode>(x, shortcut, dim_t{B, hidden});
    x = eng.insert_blend<FCNode>(x, B, 2);
    eng.dest_node = x;
    eng.finish_off();


    host_vector<float> input;
    input.resize(B * 1000);
    std::default_random_engine e;
    for(auto& x: input){
        x = 1.0 * e() % 10001 / 10000;
    }
    ForwardVisitor fwd(eng, );

}