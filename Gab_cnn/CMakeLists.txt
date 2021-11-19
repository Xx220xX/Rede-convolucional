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
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
link_libraries(OpenCL Lua543 png)

add_executable(Gab_cnn main.c cnn.h cnn.c cnn.h ponto3d.h parametros.h ${camadas} camadas/src/CamadaConv.c)
target_link_libraries(Gab_cnn tensor.dll gpu.dll lcg.dll OpenCL.dll)