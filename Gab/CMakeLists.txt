cmake_minimum_required(VERSION 3.20)
project(Gab C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS -m64)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../bin)


add_library(gabkernel SHARED project/src/cnn/kernel_lib.c project/kernels/defaultkernel.h)
set_target_properties(gabkernel PROPERTIES PREFIX "")


include_directories(external/include project/include project)
link_directories(external/lib bin)
include(project/CMakeLists.txt)

link_libraries(png OpenCL gabkernel)
link_libraries(lua543)





set(GAB_FILES_EXE ${GAB_CODE} project/include/error_list.h project/include/thread/Thread.h project/src/thread/Thread.c
        project/include/conio2/conio2.h project/src/conio2/conio2.c)
#add_executable(DEBUG_KERNEL ${KERNEL_CODE} )
#add_executable(Debug debug/dropout.c ${GAB_FILES_EXE} )

add_library(gab_library_cnn SHARED ${GAB_FILES_EXE} ${CNNLUA} project/py/cpy.c project/py/cpy.h)
link_libraries(-lshell32 -lole32 -lkernel32 -luser32 -lgdi32)
link_libraries(gdiplus)
link_libraries(Ws2_32)
add_executable(Gab main.c ui_win_api/client.h ${GAB_FILES_EXE} ${CNNLUA} project/src/setup/Setup.c project/include/setup/Setup.h)

