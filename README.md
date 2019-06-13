# Breakpoint And Recalculation Implementation
Usage: 
1. Make Sure GTX 1080 Ti or better GPU is working fine;
2. Install CUDA 9+, CMake 3.10+, gcc 7+, cudnn 7.3+ and other necessary dependencies
3. Download `cifar10/data.bin` and `cifar10/labels.bin` from http://ss.fluorinedog.com/data/net_data/ and set the filepath in data_provider.cu accordingly (Note: MNIST cannot be used unless hard-coded paramenters are modified)
4. compile the program by:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
```
5. execute `./resent/driver` to see the results. If out of memory, try less deep network(e.g. `resent50`) or lower batchsize, by changing `driver.cu`.

