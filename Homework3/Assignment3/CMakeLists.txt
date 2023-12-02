#cmake_minimum_required(VERSION 3.10)
cmake_minimum_required (VERSION 2.8.11...3.22.1)
project(Rasterizer)

set(OpenCV_DIR  "E:/library/opencv-4.8.0/build")#for find_package
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(EIGEN3_INCLUDE_DIR "C:/Program Files (x86)/Eigen3/include")
set(OpenCV_INCLUDE_DIR "E:/library/opencv-4.8.0/build/install/include")
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})



add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp Triangle.hpp Triangle.cpp global.hpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})

#link_directories("E:/library/opencv-4.8.0/build/install/x64/mingw/lib")//for no use find_package
#target_link_libraries(Rasterizer libopencv_world480.dll.a)

#target_compile_options(Rasterizer PUBLIC -Wall -Wextra -pedantic)
