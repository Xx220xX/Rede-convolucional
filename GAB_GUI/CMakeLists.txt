cmake_minimum_required(VERSION 3.21)
project(GAB_GUI C)

set(CMAKE_C_STANDARD 23)

#incluir diretorio src
include_directories(src)

# utilizar win32 api
link_libraries(-lshell32 -lole32 -lkernel32 -luser32 -lgdi32)
add_executable(GAB_GUI main.c)
