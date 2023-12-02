cmake_minimum_required (VERSION 2.8.11...3.22.1)
project (Transformation)

set(EIGEN3_INCLUDE_DIR "C:/Program Files (x86)/Eigen3/include")
#find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable (Transformation main.cpp)
