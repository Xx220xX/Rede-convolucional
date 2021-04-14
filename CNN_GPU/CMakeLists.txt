cmake_minimum_required(VERSION 3.17)
project(CNN_GPU C)

set(CMAKE_C_STANDARD 11)
link_directories(src/lua)
link_libraries(OpenCL lua543)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../python/CNN_GPU/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../build)

#add_library(CNNGPU SHARED libraryPythonWrapper.c libraryPythonWrapper.h

add_executable(CNN_GPU main2.c src/conio2/conio2.c src/conio2/conio2.h

        src/lua/lua.h src/lua/lualib.h src/lua/lauxlib.h
        src/icon/icon.o

        src/Camada.h src/Cnn.h
        src/CamadaConv.h src/Tensor.h
        src/CamadaPool.h src/CamadaDropOut.h
        src/CamadaFullConnect.h src/funcoesDeAtivacao.h
        src/gpu/Kernel.h src/gpu/Kernel.c
        src/lcg.h src/lcg.c
        src/gpu/WrapperCL.h src/gpu/WrapperCL.c
        )