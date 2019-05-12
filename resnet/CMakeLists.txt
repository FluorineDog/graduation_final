add_executable(resnet resnet.cu)
target_link_libraries(resnet cublas cudnn)
add_library(cu_extra cu_extra.cu)
target_link_libraries(resnet cu_extra)