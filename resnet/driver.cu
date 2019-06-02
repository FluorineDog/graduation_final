#include "helper/common.h"
#include "../doglib/time/timer.h"
#include "data_provider.h"
#include "engine.h"
#include "resnet.h"

Global global;
int main() {
    Engine eng;
    // define network structure
    int B = 256;
    int pixel = 32;
    int features = 3 * pixel * pixel;
    int classes = 10;
    dim_t input_dim = {B, 3, pixel, pixel};

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
        timer.reset();
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
        auto init_tm = timer.get_step_seconds();
        eng.zero_grad();
        auto zero_tm = timer.get_step_seconds();
        eng.forward_pass(data_beg);
        auto fwd_tm = timer.get_step_seconds();
        auto act = eng.get_dest_feature();
        auto act_grad = eng.get_dest_gradient();
        device_vector<int> dev_labels(labels_beg, labels_end);
        // dog_print("##", act, dim_t{B, classes});
        ce.forward(losses, act, dev_labels.data().get());
        // eng.get_mm().l2_forward(losses, B, 0.1);
        // dog_print("??", losses, dim_t{B});
        auto loss = thrust::reduce(thrust::device, losses.begin(), losses.end());

        double ck_tm = 0;
        if(offset_lb) {
            ce.backward(act_grad, act, losses, dev_labels.data().get());
            ck_tm = timer.get_step_seconds();
            eng.backward_pass(act_grad);
        }
        auto back_tm = timer.get_step_seconds();
        auto correct = get_acc(act, labels_beg, B, classes);
        auto acc_tm = timer.get_step_seconds();
        if(loss != loss) {
            break;
        }
        if(offset_lb) {
            static float lr = 0.0002 / B;
            eng.get_opt().step(lr);
            auto all_tm = timer.get_overall_seconds();
            cout << loss / B << " " << correct << " " << endl;
            cout << "init_tm"
                 << " " << init_tm << " "
                 << "zero_tm"
                 << " " << zero_tm << " "
                 << "fwd_tm"
                 << " " << fwd_tm << " "
                 << "cross_tm"
                 << " " << ck_tm << " "
                 << "bak_tm"
                 << " " << back_tm << " "
                 << "acc_tm"
                 << " " << acc_tm << " "
                 << "all_tm"
                 << " " << all_tm << " "    //
                ;
        } else {
            auto t = timer.get_step_seconds();
            cout << "test: " << loss / B << " " << correct << " " << t << endl;
        }
        cout << endl;
    }
}