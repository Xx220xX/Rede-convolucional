cmake_minimum_required(VERSION 3.15)
project(gab C)
link_directories(lib)
link_libraries(OpenCL)
set(CMAKE_C_STANDARD 11)

#add_executable(gab main.c
        add_library(gab SHARED
        gabriela_gpu.c gabriela_gpu.h
#        src/gabriela.c src/gabriela.h
        src/WrapperCL.c src/WrapperCL.h src/Kernel.c src/Kernel.h src/matGPU.c src/matGPU.h src/lcg.c src/lcg.h
        src/gabrielaLayer.c src/gabrielaLayer.h
                )