cmake_minimum_required(VERSION 3.17)
project(CNN_GPU C)

set(CMAKE_C_STANDARD 11)

add_executable(CNN_GPU main.c
        src/Camada.h src/Cnn.h
        src/CamadaConv.h src/Tensor.h
        src/CamadaPool.h src/CamadaDropOut.h
        src/CamadaFullConnect.h src/funcoesDeAtivacao.h
        src/gpu/Kernel.h src/gpu/Kernel.c
        src/gpu/lcg.h src/gpu/lcg.c
        src/gpu/matGPU.h src/gpu/matGPU.c
        src/gpu/WrapperCL.h src/gpu/WrapperCL.c
        )