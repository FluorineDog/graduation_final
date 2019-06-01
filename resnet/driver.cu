#include "../doglib/time/timer.h"
#include "stdafx.h"
#include "computational_graph.h"
#include "resnet.h"

Global global;
int main() {
    Engine eng;
    // define network structure
    int B = 200;
    int features = 28 * 28;
    int classes = 10;
    dim_t input_dim = {B, 1, 28, 28};

    auto x = eng.insert_leaf<PlaceHolderNode>(input_dim);
    eng.src_node = x;
    // x = construct_resnet(eng, x, {1, 1, 1, 1}, classes);
    x = resnet50(eng, x, classes);
    eng.dest_node = x;
    eng.finish_off();

    auto total = 60000;
    host_vector<float> data_raw = get_data();
    host_vector<int> labels_raw = get_labels();

    cout << endl;

    DeviceVector<T> losses(B);
    CrossEntropy ce(B, classes);
    global.update_workspace_size(ce.workspace());
    doglib::time::TimerAdvanced timer([] { cudaDeviceSynchronize(); });
    for(auto x : Range(300)) {
        auto offset_lb = x % (total / B) * B;
        // offset_lb = 0;
        auto offset_dt = offset_lb * features;

        if(offset_lb != 0) {
            global.set_training(true);
        } else {
            global.set_training(false);
        }

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

        if(offset_lb) {
            ce.backward(act_grad, act, losses, dev_labels.data().get());
            eng.backward_pass(act_grad);
        }
        auto correct = get_acc(act, labels_beg, B, classes);
        if(loss != loss) {
            break;
        }
        if(offset_lb) {
            static float lr = 0.0002 / B;
            eng.get_opt().step(lr);
            auto t = timer.get_step_seconds();
            cout << loss / B << " " << correct << " " << t << endl;
        } else {
            auto t = timer.get_step_seconds();
            cout << "test: " << loss / B << " " << correct << " " << t << endl;
        }
    }
}