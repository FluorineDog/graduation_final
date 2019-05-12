__global__ void add_biased_impl(float *out, int len, int total_size, float alpha, float *in){
    auto id = threadIdx.x + blockIdx.x * blockDim.x; 
    if(id < total_size){
        auto k_id = id % len; 
        out[id] = alpha * in[k_id];
    }
}

void add_biased(float* out, int len, int batch,  float alpha, float* in){
    int size = batch*len;
    add_biased_impl<<<(size + 255) / 256, 256>>>(out, len, size, alpha, in);
}