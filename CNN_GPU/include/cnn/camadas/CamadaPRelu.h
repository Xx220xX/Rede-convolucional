//
// Created by gf154 on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_PRELU_H
#define CNN_GPU_CAMADA_PRELU_H

#include "Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>

typedef struct {
	Typecamada super;
	Tensor A;
	Tensor dA;
	char ramdomType;
	Kernel kernelPReluAtiva;
	Kernel kernelPReluCalcGrads;
} *CamadaPRelu, TypecamadaPRelu;


Camada createPRelu(WrapperCL *cl, QUEUE queue, unsigned int inx, unsigned int iny,
				   unsigned int inz, Tensor entrada,
				   Params params,
				   RandomParam  randomParams, CNN_ERROR *error);


#endif //CNN_GPU_CAMADA_PRelu_H
