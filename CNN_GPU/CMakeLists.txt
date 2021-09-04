cmake_minimum_required(VERSION 3.17)
project(CNN_GPU C)

set(CMAKE_C_FLAGS -m64)
set(CMAKE_C_STANDARD 11)
link_directories(lib/lua)
include_directories(include include/cnn)

link_libraries(OpenCL lua543)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(lua include/lua/lua.h include/lua/lualib.h include/lua/lauxlib.h)

set(conio2 src/conio2/conio2.c include/conio2/conio2.h)

set(LCG_RANDOM include/cnn/LCG_Random/lcg.h src/LCG_Random/lcg.c)

set(camadasCNN include/cnn/camadas/Camada.h src/camadas/Camada.c
        include/cnn/camadas/CamadaConv.h src/camadas/CamadaConv.c
        include/cnn/camadas/CamadaConvNC.h src/camadas/CamadaConvNC.c
        include/cnn/camadas/CamadaPool.h src/camadas/CamadaPool.c
        include/cnn/camadas/CamadaPoolAv.h src/camadas/CamadaPoolAv.c
        include/cnn/camadas/CamadaDropOut.h src/camadas/CamadaDropOut.c
        include/cnn/camadas/CamadaFullConnect.h src/camadas/CamadaFullConnect.c
        include/cnn/camadas/CamadaBatchNorm.h src/camadas/CamadaBatchNorm.c
        include/cnn/camadas/CamadaRelu.h src/camadas/CamadaRelu.c
        include/cnn/camadas/CamadaPadding.h src/camadas/CamadaPadding.c
        include/cnn/camadas/CamadaSoftMax.h src/camadas/CamadaSoftMax.c
        include/cnn/camadas/funcoesDeAtivacao.h
        )
set(cnn include/cnn/cnn.h src/cnn.c src/CnnLua.h
        include/cnn/cnn_errors_list.h

        include/cnn/utils/vectorUtils.h src/utils/vectorUtils.c
        include/cnn/utils/manageTrain.h src/utils/manageTrain.c

        )
set(time_util src/utils/time_utils.c include/cnn/utils/time_utils.h)
set(list_util src/utils/list_args.c include/cnn/utils/list_args.h)
set(dir_util src/utils/dir.c include/cnn/utils/dir.h)
set(string_util src/utils/String.c include/cnn/utils/String.h)
set(default_kernel_util src/utils/defaultkernel.c include/cnn/utils/defaultkernel.h)
set(thread_util include/cnn/utils/Thread.h)
set(memory_util include/cnn/utils/memory_utils.h src/utils/memory_utils.c)
set(tensor include/cnn/tensor/Tensor.h src/tensor/Tensor.c)

set(gpu include/cnn/gpu/Kernel.h src/gpu/Kernel.c include/cnn/gpu/WrapperCL.h src/gpu/WrapperCL.c)

set(CNNpython src/libraryPythonWrapper.c include/cnn/libraryPythonWrapper.h)

set(kernels kernels/camadas/utils.h kernels/camadas/bathnorm.h kernels/camadas/conv.h
        kernels/camadas/dropout.h kernels/camadas/fullconnect.h kernels/camadas/pool.h
        kernels/camadas/relu.h kernels/camadas/softmax.h
        kernels/camadas/cnnutils.h)

#add_library(CNNGPU SHARED
#        ${CNNpython}
#        src/icon/icon.o
#        ${cnn}
#        #        ${kernels}
#        ${time_util}
#        ${list_util}
#        ${dir_util}
#        ${string_util}
#        ${default_kernel_util}
#        ${thread_util}
#        ${gpu}
#        ${camadasCNN}
#        ${memory_util}
#        ${LCG_RANDOM}
#        ${tensor}
#        ${conio2} include/cnn/config.h)

add_executable(CNN_GPU main.c ${lua}
        src/icon/icon.o
        ${cnn}
        ${time_util}
        ${list_util}
        ${dir_util}
        ${string_util}
        ${default_kernel_util}
        ${thread_util}
        ${gpu}
        ${camadasCNN}
        ${memory_util}
        ${LCG_RANDOM}
        ${tensor}
        ${conio2} include/cnn/config.h)

#include(${CMAKE_CURRENT_SOURCE_DIR}/teste/CMakeTensor.txt)
