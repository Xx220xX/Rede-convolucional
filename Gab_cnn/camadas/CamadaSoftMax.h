//
// Created by Henrique on 5/8/2021.
//

#ifndef CNN_GPU_CAMADASOFTMAX_H
#define CNN_GPU_CAMADASOFTMAX_H

#include "camada.h"


typedef struct {
	Camada_t super;
	Kernel kernelSoftMaxAtiva1;
	Kernel kernelSoftMaxAtiva2;
	Kernel kernelSoftMaxAtiva3;
	Kernel kernelSoftMaxCalcGrads;
	Tensor soma;
	Tensor exponent;

} *CamadaSoftMax, TypecamadaSoftMax;


Camada createSoftMax(WrapperCL *cl, QUEUE queue, unsigned int inx, unsigned int iny,
					 unsigned int inz, Tensor entrada,
					  CNN_ERROR *error);


#endif //CNN_GPU_CAMADASOFTMAX_H
