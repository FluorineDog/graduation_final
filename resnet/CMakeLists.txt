add_executable(resnet resnet.cu)
add_executable(driver driver.cu)
file(GLOB headers *.h)
add_library(cross_entropy cross_entropy.cu cg.cu)

target_link_libraries(resnet cublas cudnn)
target_link_libraries(resnet cross_entropy)

target_link_libraries(driver cublas cudnn)
target_link_libraries(driver cross_entropy)
