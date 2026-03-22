cmake_minimum_required(VERSION 3.13.3)
project(mapping)

list(APPEND CMAKE_PREFIX_PATH "/home/leon/anaconda3/envs/seg37/lib/python3.7/site-packages/pybind11/share/cmake/pybind11")


set (CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "/home/leon/Documents/Projects/MASS/mapping/")  # 修改过
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIRS}) 

include_directories(/home/leon/Documents/Projects/eigen-linux/include/eigen3)

#include_directories("/home/leon/Documents/Projects/MASS/mapping/mapping.cpp") # 修改过

# add_subdirectory(pybind11)
find_package(pybind11)
pybind11_add_module(mapping mapping.cpp)

# find_package(OpenMP)
# target_link_libraries(mapping PRIVATE OpenMP::OpenMP_CXX)
