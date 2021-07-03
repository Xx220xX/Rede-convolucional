cmake_minimum_required(VERSION 3.17)
project(CNN_GPU C)

set(CMAKE_C_FLAGS -m64)
set(CMAKE_C_STANDARD 11)
link_directories(src/lua src/SDL2/x64/lib)
include_directories(src/SDL2/x64/include)

#link_libraries(SDL2.dll SDL2 SDL2main )
link_libraries(OpenCL lua543)
#link_libraries(python3 python3_d python39 python39_d)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(lua src/lua/lua.h src/lua/lualib.h src/lua/lauxlib.h)
set(conio2 src/conio2/conio2.c src/conio2/conio2.h)
set(LCG_RANDOM src/LCG_Random/lcg.h src/LCG_Random/lcg.c)
set(camadasCNN src/camadas/Camada.h src/camadas/Camada.c
        src/camadas/CamadaConv.h src/camadas/CamadaConv.c
        src/camadas/CamadaPool.h src/camadas/CamadaPool.c
        src/camadas/CamadaDropOut.h src/camadas/CamadaDropOut.c
        src/camadas/CamadaFullConnect.h src/camadas/CamadaFullConnect.c
        src/camadas/CamadaBatchNorm.h src/camadas/CamadaBatchNorm.c
        src/camadas/CamadaRelu.h src/camadas/CamadaRelu.c
        src/camadas/CamadaPadding.h src/camadas/CamadaPadding.c
        src/camadas/CamadaSoftMax.c src/camadas/CamadaSoftMax.h
        src/camadas/funcoesDeAtivacao.h
        src/Cnn.h src/cnn.c)
set(tensor src/tensor/Tensor.h src/tensor/Tensor.c)
set(gpu src/gpu/Kernel.h src/gpu/Kernel.c src/gpu/WrapperCL.h src/gpu/WrapperCL.c)
set(CNNpython src/libraryPythonWrapper.c src/libraryPythonWrapper.h )
set(kernels kernels/camadas/utils.h kernels/camadas/bathnorm.h kernels/camadas/conv.h
        kernels/camadas/dropout.h kernels/camadas/fullconnect.h kernels/camadas/pool.h
        kernels/camadas/relu.h kernels/camadas/softmax.h
        )
#add_library(CNNGPU SHARED ${CNNpython}
add_executable(CNN_GPU main.c ${lua}
#add_executable(testeALLCNN debug/testeAllCNN.c
#        ${kernels}
        src/defaultkernel.h
        src/icon/icon.o
        ${gpu}
        ${camadasCNN}
        ${LCG_RANDOM}
        ${tensor}
        ${conio2})
#add_executable(testeHUD src/treino/sdlHUD.c)

#add_executable(TESTE_OPENCL testeOpenCL.c src/gpu/WrapperCL.h src/gpu/WrapperCL.c src/gpu/Kernel.h src/gpu/Kernel.c)