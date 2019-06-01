#include "helper/common.h"
#include <fstream>
#include <arpa/inet.h>
#include <thrust/device_vector.h>
using std::vector;
// const char* data_file = "/home/guilin/workspace/data/mnist/images-idx3-ubyte";
// const char* labels_file = "/home/guilin/workspace/data/mnist/labels-idx1-ubyte";

const char* data_file = "/home/guilin/workspace/data/cifar10/data.bin";
const char* labels_file = "/home/guilin/workspace/data/mnist/labels.bin";

host_vector<float> get_data() {
    host_vector<float> data;
    std::ifstream fin(data_file, std::ios::binary);
    // uint32_t magic;
    uint32_t number = 60000, w = 32, h = 32;
    // fin.read((char*)&magic, 4);
    // fin.read((char*)&number, 4);
    // fin.read((char*)&w, 4);
    // fin.read((char*)&h, 4);
    // magic = htonl(magic);
    // number = htonl(number);
    // h = htonl(h);
    // w = htonl(w);
    // assert(magic == 0x00000803);
    assert(number == 60000);
    assert(h == 32);
    assert(w == 32);
    number = 60000;
    auto sz = number * w * h * 3;
    data.resize(sz);
    vector<uint8_t> buffer(sz);
    fin.read((char*)buffer.data(), sz);
    #pragma omp parallel for
    for(size_t i = 0; i < sz; ++i) {
        auto id = i;
        uint8_t x = buffer[id];
        assert(0 <= x && x < 256);
        data[id] = x / 255.0;
    }
    return data;
}
host_vector<int> get_labels() {
    host_vector<int> data;
    std::ifstream fin(labels_file, std::ios::binary);
    uint32_t magic;
    uint32_t number = 60000;
    // fin.read((char*)&magic, 4);
    // fin.read((char*)&number, 4);
    // magic = htonl(magic);
    // number = htonl(number);
    // assert(magic == 0x00000801);
    assert(number == 60000);
    auto sz = number;
    data.resize(sz);
    vector<uint8_t> buffer(sz);
    fin.read((char*)buffer.data(), sz);
    for(auto id : Range(sz)) {
        uint8_t x = buffer[id];
        // assert(0 <= x && x < 10);
        data[id] = x;
    }
    return data;
}

float get_acc(float* dev_logits, int* labels, int N, int feature) {
    vector<float> buffer(N * feature);
    cudaMemcpy(buffer.data(), dev_logits, N * feature * sizeof(float), cudaMemcpyDefault);
    int count = 0;
    for(int b : Range(N)) {
        auto loc = std::max_element(
                       buffer.begin() + b * feature, buffer.begin() + (b + 1) * feature) -
                   buffer.begin() - b * feature;
        //
        assert(0 <= loc && loc < feature);
        assert(0 <= loc && loc < feature);
        count += (loc == labels[b]) ? 1 : 0;
    }
    return count * 1.0 / N;
}
