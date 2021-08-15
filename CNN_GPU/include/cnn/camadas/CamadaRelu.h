//
// Created by gf154 on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_RELU_H
#define CNN_GPU_CAMADA_RELU_H

#include "Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>

#ifndef UINT
//#define UINT unsigned int
#endif
typedef struct {
	Typecamada super;
	Kernel kernelReluAtiva;
	Kernel kernelReluCalcGrads;
} *CamadaRelu, TypecamadaRelu;




Camada createRelu(WrapperCL *cl, QUEUE queue, unsigned int inx, unsigned int iny,
				  unsigned int inz, Tensor entrada,
				  char usehost, CNN_ERROR *error);


#endif //CNN_GPU_CAMADA_RELU_H
