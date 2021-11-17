//
// Created by gf154 on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_PRELU_H
#define CNN_GPU_CAMADA_PRELU_H

#include "camada.h"


typedef struct CamadaPRelu_t {
	Camada_t super;
	Tensor A;
	Tensor dA;
	Kernel kernelPReluAtiva;
	Kernel kernelPReluCalcGrads;
} *CamadaPRelu, CamadaPRelu_t;

Camada createPRelu(WrapperCL *cl, QUEUE queue, Ponto3d size_in, Tensor entrada, Params params, RandomParam randomParams);


#endif //CNN_GPU_CAMADA_PRelu_H
