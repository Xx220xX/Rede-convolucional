//
// Created by Henrique on 03/08/2021.
//

#ifndef CNN_GPU_CAMADA_Padding_H
#define CNN_GPU_CAMADA_Padding_H

#include "camada.h"

typedef struct CamadaPadding_t {
	Camada_t super;
	size_t top, bottom, left, right;
	Kernel ativa, calcGrad;
} *CamadaPadding, CamadaPadding_t;

extern Camada createPadding(Gpu gpu, QUEUE queue, P3d size_in, size_t top, size_t bottom, size_t left, size_t right, Tensor entrada);


#endif //CNN_GPU_CAMADA_Padding_H
