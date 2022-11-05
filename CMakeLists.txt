cmake_minimum_required(VERSION 3.17)
project(NN)

set(CMAKE_CXX_STANDARD 17)

SET(GCC_COVERAGE_COMPILE_FLAGS "-Ofast -fopenmp")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

add_executable(NN main.cpp src/data.cpp src/net.h src/activ.h src/optim.h src/layer.h src/activ.cpp src/optim.cpp src/init.h src/net.cpp src/utils.cpp src/utils.h src/data.h src/trainer.h)