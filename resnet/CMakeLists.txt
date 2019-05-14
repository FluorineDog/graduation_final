add_executable(resnet resnet.cu)
target_link_libraries(resnet cublas cudnn)
add_library(cross_entropy cross_entropy.cu)
target_link_libraries(resnet cross_entropy)