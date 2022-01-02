cmake_minimum_required(VERSION 3.20)

set(CAMADA
        project/include/camadas/camada.h
        project/include/camadas/CamadaBatchNorm.h
        project/include/camadas/CamadaConv.h
        project/include/camadas/CamadaConvF.h
        project/include/camadas/CamadaConv2D.h
        project/include/camadas/CamadaConvNC.h
        project/include/camadas/CamadaDropOut.h
        project/include/camadas/CamadaFullConnect.h
        project/include/camadas/CamadaPadding.h
        project/include/camadas/CamadaPool.h
        project/include/camadas/CamadaPRelu.h
        project/include/camadas/CamadaRelu.h
        project/include/camadas/CamadaSoftMax.h
        project/include/camadas/funcoesDeAtivacao.h

        project/src/camadas/camada.c
        project/src/camadas/CamadaBatchNorm.c
        project/src/camadas/CamadaConv.c
        project/src/camadas/CamadaConvF.c
        project/src/camadas/CamadaConv2D.c
        project/src/camadas/CamadaConvNC.c
        project/src/camadas/CamadaDropOut.c
        project/src/camadas/CamadaFullConnect.c
        project/src/camadas/CamadaPadding.c
        project/src/camadas/CamadaPool.c
        project/src/camadas/CamadaPRelu.c
        project/src/camadas/CamadaRelu.c
        project/src/camadas/CamadaSoftMax.c
        )

set(CNN
        project/include/cnn/cnn.h
        project/include/cnn/parametros.h
        project/include/cnn/ponto3d.h
        project/src/cnn/cnn.c)
set(CNNLUA
        project/include/cnn/cnn_lua.h
        project/src/cnn/cnn_lua.c
        project/src/cnn/cnn_lua_console.c)
set(KERNELDLL project/include/cnn/kernel_lib.h project/src/cnn/kernel_lib.c)

set(GPU
        project/include/gpu/Gpu.h
        project/include/gpu/Kernel.h
        project/src/gpu/Gpu.c
        project/src/gpu/Kernel.c)
set(LCG project/include/lcg/lcg.h project/src/lcg/lcg.c)
set(TENSOR
        project/include/tensor/exc.h
        project/include/tensor/tensor.h
        project/include/tensor/tensor_flags.h
        project/include/config.h
        project/src/tensor/exc.c
        project/src/tensor/tensor.c
        )
set(KERNEL_CAMADAS project/kernels/camadas/bathnorm.h
        project/kernels/camadas/cnnutils.h
        project/kernels/camadas/conv.h
        project/kernels/camadas/convf.h
        project/kernels/camadas/convNc.h
        project/kernels/camadas/dropout.h
        project/kernels/camadas/fullconnect.h
        project/kernels/camadas/padding.h
        project/kernels/camadas/poolav.h
        project/kernels/camadas/poolMax.h
        project/kernels/camadas/poolMin.h
        project/kernels/camadas/prelu.h
        project/kernels/camadas/relu.h
        project/kernels/camadas/softmax.h
        project/kernels/camadas/utils.h
        )
set(KERNEL_CODE ${KERNEL_CAMADAS}
        project/kernels/debugKernel.c)
set(GAB_CODE ${CNN} ${TENSOR} ${LCG} ${GPU} ${CAMADA})

