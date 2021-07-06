//
// Created by Henrique on 5/8/2021.
//

#ifndef CNN_GPU_CAMADASOFTMAX_H
#define CNN_GPU_CAMADASOFTMAX_H

#include "Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>

#define UINT unsigned int
typedef struct {
	Typecamada super;
	Kernel kernelSoftMaxAtiva1;
	Kernel kernelSoftMaxAtiva2;
	Kernel kernelSoftMaxCalcGrads;
	Tensor soma;
	Tensor exponent;

} *CamadaSoftMax, TypecamadaSoftMax;

void realeaseSoftMax(CamadaSoftMax *pc);

void ativaSoftMax(CamadaSoftMax c);

void corrige_pesosSoftMax(CamadaSoftMax);

void calc_gradsSoftMax(CamadaSoftMax c, Tensor GradNext);

void salvarSoftMax(WrapperCL *cl, CamadaSoftMax c, FILE *dst, GPU_ERROR *error);

Camada createSoftMax(WrapperCL *cl, QUEUE queue, unsigned int inx, unsigned int iny,
                     unsigned int inz, Tensor entrada, GPU_ERROR *error);


#endif //CNN_GPU_CAMADASOFTMAX_H
