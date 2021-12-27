//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONV_H
#define CNN_GPU_CAMADACONV_H

#include "string.h"
#include"camada.h"
#include <stdlib.h>


typedef struct CamadaConv_t {
	Camada_t super;
	Tensor W;
	Tensor dW;
	size_t passox, passoy;
	Kernel convSum;
	Kernel convCalcGradAndFixWeight;
	Kernel convCalcGradBatch;
	Kernel kernel_fixW;
	Kernel convCalcGradIn;
	RandomParams rdp_filtros;
} *CamadaConv, CamadaConv_t;

extern Camada CamadaConv_new(INTERNAL_DEFAULT_ARGS, P2d passo, P3d filtro, Parametros params, RandomParams rdp_filtros);

extern Camada CamadaConv_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADACONV_H
