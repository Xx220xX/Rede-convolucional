cmake_minimum_required(VERSION 3.20)
project(ui)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS -m64)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../bin)


include_directories(SDL/include)
link_directories(SDL/lib)
link_libraries(-lshell32 -lole32 -lkernel32 -luser32 -lgdi32)
link_libraries(
        SDL2main
        SDL2
        SDL2_image
        SDL2_test
        )
#link_libraries(gdiplus)
add_executable(ui main.c)
