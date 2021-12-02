//
// Created by Henrique on 03/08/2021.
//

#ifndef CNN_GPU_CAMADA_Padding_H
#define CNN_GPU_CAMADA_Padding_H

#include "Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>

typedef struct {
	Typecamada super;
	size_t top, bottom, left, right;
	Kernel ativa, calcGrad;
} *CamadaPadding, TypecamadaPadding;

Camada createPadding(WrapperCL *cl, QUEUE queue,
					 UINT inx, UINT iny, UINT inz,
					 UINT top, UINT bottom, UINT left, UINT right, Tensor entrada,
					  CNN_ERROR *error);



#endif //CNN_GPU_CAMADA_Padding_H
