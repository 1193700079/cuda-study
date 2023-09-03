cmake_minimum_required(VERSION 3.10)

project(cpp_study)  # 将"YourProjectName"替换为你的项目名称

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# 添加可执行文件
add_executable(cpp_study  cpp/01.cpp)  #
add_executable(cpp_study2  cpp/02.cpp)  #
add_executable(cpp_study3  cpp/03refer.cpp)  #
add_executable(cuda_study  3_test.cu)  #

target_link_libraries(cuda_study  ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})