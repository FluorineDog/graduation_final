add_executable(resnet resnet.cu)
target_link_libraries(resnet cublas cudnn)
