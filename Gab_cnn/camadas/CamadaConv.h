
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
	Tensor filtros;
	Tensor grad_filtros;
	size_t passox, passoy;
	Kernel convSum;
	Kernel convCalcGradAndFixWeight;
	Kernel convCalcGradIn;
	RandomParams rdp_filtros;
} *CamadaConv, CamadaConv_t;

extern Camada CamadaConv_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, Tensor entrada,
							 Parametros params, Ecx ecx, RandomParams rdp_filtros);


#endif //CNN_GPU_CAMADACONV_H
