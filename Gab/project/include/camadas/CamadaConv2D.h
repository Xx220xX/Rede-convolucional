//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONV2D_H
#define CNN_GPU_CAMADACONV2D_H

#include "string.h"
#include"camada.h"
#include "funcoesDeAtivacao.h"

typedef struct CamadaConv2D_t {
	Camada_t super;
	Tensor W;
	Tensor z;
	Tensor dz;
	Tensor dW;
	size_t passox, passoy;
	Kernel conv2dSum;
	Kernel conv2dCalcGradZ;
	Kernel conv2dCalcGradAndFixWeight;
	Kernel conv2dCalcGradIn;
	Kernel conv2dCalcGradBatch;
	Kernel kernel_fixW;
	uint32_t fid;
	uint32_t dfid;
	RandomParams rdp_filtros;
} *CamadaConv2D, CamadaConv2D_t;

extern Camada CamadaConv2D_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, uint32_t fid, Tensor entrada, Parametros params, Ecx ecx, RandomParams rdp_filtros);

extern Camada CamadaConv2D_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADACONV2D_H
