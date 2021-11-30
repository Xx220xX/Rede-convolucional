cmake_minimum_required(VERSION 3.20)
project(tensor C)
set(CMAKE_C_FLAGS -m64)
set(CMAKE_C_STANDARD 11)
link_directories(lib)
include_directories(include)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

add_library(lcg STATIC include/config.h include/lcg/lcg.h include/lcg/lcg.c)
#set_target_properties(lcg PROPERTIES PREFIX "")

link_libraries(png OpenCL )
add_library(gpu STATIC include/config.h include/gpu/Gpu.h include/gpu/Gpu.c include/gpu/kernel.c include/gpu/kernel.h)
#set_target_properties(gpu PROPERTIES PREFIX "")

link_libraries(gpu)
add_library(tensor STATIC include/config.h include/tensor.c include/tensor/tensor.h include/tensor/tensor_flags.h include/png/png.h include/lcg/lcg.h include/tensor/exc.c include/tensor/exc.h  include/tensor/exc.h include/tensor/exc.c)
#set_target_properties(tensor PROPERTIES PREFIX "")
#target_link_libraries(tensor gpu.dll lcg.dll  OpenCL.dll )
link_libraries(tensor)
link_libraries(lcg)
add_executable(stack main_stack.c include/tensor/exc.h include/tensor/exc.c)

add_executable(gpuTeste main_gpu.c)
add_executable(tensor_test main.c )

