cmake_minimum_required(VERSION 3.20)
project(Gab_cnn C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS -m64)

include_directories(extern/include ./)
link_directories(extern/lib bin)

set(camadas
        camadas/src/camada.c
        camadas/CamadaBatchNorm.h
        camadas/CamadaConv.h
        camadas/CamadaConvF.h
        camadas/CamadaConvNC.h
        camadas/CamadaDropOut.h
        camadas/CamadaFullConnect.h
        camadas/CamadaPadding.h
        camadas/CamadaPool.h
        camadas/CamadaPRelu.h
        camadas/CamadaRelu.h
        camadas/CamadaSoftMax.h
        camadas/funcoesDeAtivacao.h
        camadas/src/camada.c
        camadas/src/CamadaBatchNorm.c
        camadas/src/CamadaConv.c
        camadas/src/CamadaConvF.c
        camadas/src/CamadaConvNC.c
        camadas/src/CamadaDropOut.c
        camadas/src/CamadaFullConnect.c
        camadas/src/CamadaPadding.c
        camadas/src/CamadaPool.c
        camadas/src/CamadaPRelu.c
        camadas/src/CamadaRelu.c
        camadas/src/CamadaSoftMax.c)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../bin)
link_libraries(Lua543 png)


add_library(gabkernel SHARED cnn/src/kernel_lib.c cnn/src/kernel_lib.c kernels/defaultkernel.h)
set_target_properties(gabkernel PROPERTIES PREFIX "")

link_libraries(tensor lcg gpu OpenCL)
set(CNN cnn/ponto3d.h cnn/parametros.h ${camadas} cnn/cnn.h cnn/src/cnn.c cnn/cnn_lua.h cnn/src/cnn_lua.c
        cnn/src/cnn_lua_console.c
        extern/include/conio2/conio2.c extern/include/conio2/conio2.h
        )
set(TREINO treino/manageTrain.c
        treino/Manage.h
        treino/Thread.c
        treino/Thread.h
        treino/dir.h
        treino/dir.c)
add_executable(Gab_cnn main.c ${CNN} ${TREINO})
add_library(gabCnn STATIC ${CNN})
set_target_properties(gabCnn PROPERTIES PREFIX "")

target_link_libraries(Gab_cnn gabkernel.dll)
target_link_libraries(gabCnn gabkernel.dll)



add_executable(Kernels kernels/camadas/bathnorm.h
        kernels/camadas/cnnutils.h
        kernels/camadas/conv.h
        kernels/camadas/convf.h
        kernels/camadas/convNc.h
        kernels/camadas/dropout.h
        kernels/camadas/fullconnect.h
        kernels/camadas/padding.h
        kernels/camadas/poolav.h
        kernels/camadas/poolMax.h
        kernels/camadas/poolMin.h
        kernels/camadas/prelu.h
        kernels/camadas/relu.h
        kernels/camadas/softmax.h
        kernels/camadas/utils.h kernels/debugKernel.c)