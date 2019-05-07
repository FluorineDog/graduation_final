add_executable(resnet resnet.cpp)
cuda_add_library(resnet_cu resnet_cu.cu)
target_link_libraries(resnet_cu cublas cudnn)
target_link_libraries(resnet resnet_cu)
