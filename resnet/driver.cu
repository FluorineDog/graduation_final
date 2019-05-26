#include "stdafx.h"

Global global;
int main() {
    Engine eng;
    // define network structure
    int B = 200;
    int features = 28 * 28;
    int hidden = 1000;
    int classes = 10;
    dim_t input_dim = {B, features};

    auto x = eng.insert_leaf<PlaceHolderNode>(input_dim);
    eng.src_node = x;

    // auto shortcut = x;
    x = eng.insert_node<FCNode>(x, B, features, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_node<FCNode>(x, B, hidden, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    x = eng.insert_node<FCNode>(x, B, hidden, hidden);
    x = eng.insert_node<ActivationNode>(x, dim_t{B, hidden});
    // x = eng.insert_blend<AddNode>(x, shortcut, dim_t{B, hidden});

    x = eng.insert_node<FCNode>(x, B, hidden, classes);
    eng.dest_node = x;
    eng.finish_off();

    auto total = 60000;
    host_vector<float> data_raw = get_data();
    host_vector<int> labels_raw = get_labels();

    // auto total = B;
    // host_vector<float> data_raw;
    // host_vector<int> labels_raw;
    // data_raw.resize(B * 1000);
    // std::default_random_engine e(2);
    // for(auto& x : data_raw) {
    //     x = (float)(e() % 10001) / 5000 - 1;
    // }
    // for(auto id : Range(B)) {
    //     float sum = 0;
    //     for(auto x : Range(features)) {
    //         sum += data_raw[id * features + x];
    //     }
    //     int label = sum >= 0 ? 1 : 0;
    //     labels_raw.push_back(label);
    // }

    // for(auto x: labels_raw){
    //     cout << x;
    // }
    cout << endl;

    DeviceVector<T> losses(B);
    CrossEntropy ce(B, classes);
    global.update_workspace_size(ce.workspace());
    for(auto x : Range(20000)) {
        auto offset_lb = x % (total / B) * B;
        // offset_lb = 0;
        auto offset_dt = offset_lb * features;
        auto data_beg = data_raw.data() + offset_dt;
        auto data_end = data_raw.data() + offset_dt + B * features;
        auto labels_beg = labels_raw.data() + offset_lb;
        auto labels_end = labels_raw.data() + offset_lb + B;
        eng.zero_grad();
        eng.forward_pass(data_beg);
        auto act = eng.get_ptr(eng.dest_node);
        auto act_grad = eng.get_ptr(~eng.dest_node);
        device_vector<int> dev_labels(labels_beg, labels_end);
        // dog_print("##", act, dim_t{B, classes});
        ce.forward(losses, act, dev_labels.data().get());
        // eng.get_mm().l2_forward(losses, B, 0.1);
        // dog_print("??", losses, dim_t{B});
        auto loss = thrust::reduce(thrust::device, losses.begin(), losses.end());

        // eng.get_mm().l2_backward(losses, B, 0.1);
        ce.backward(act_grad, act, losses, dev_labels.data().get());
        // dog_print("SS", act_grad, dim_t{B, classes});
        // dog_print("hhd", act, {B});

        eng.backward_pass(act_grad);
        // auto correct = thrust::count_if(losses.begin(), losses.end(), functor());
        auto correct = get_acc(act, labels_beg, B, classes);
        if(loss != loss) {
            break;
        }
        if(offset_lb) {
            eng.get_opt().step(0.001/B);
            cout << loss / B << " " << correct << endl;
        } else {
            cout << "test: " << loss / B << " " << correct << endl;
        }
    }
}