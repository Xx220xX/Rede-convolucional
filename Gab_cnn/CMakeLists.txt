cmake_minimum_required(VERSION 3.20)
project(Gab_cnn C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS -m64)

include_directories(extern/include)
link_directories(extern/lib)

link_libraries(tensor lcg gpu   OpenCL Lua543 png)

add_executable(Gab_cnn main.c)
