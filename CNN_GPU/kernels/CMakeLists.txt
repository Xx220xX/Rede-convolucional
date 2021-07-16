cmake_minimum_required(VERSION 3.17)
project(CNN_GPU C)

set(CMAKE_C_FLAGS -m64)
set(CMAKE_C_STANDARD 11)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(kernels camadas/utils.h camadas/bathnorm.h camadas/conv.h
        camadas/dropout.h camadas/fullconnect.h camadas/pool.h
        camadas/relu.h camadas/softmax.h
        )
#add_library(CNNGPU SHARED ${CNNpython}
add_executable(CNN_GPU main.c )