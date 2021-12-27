
//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONVF_H
#define CNN_GPU_CAMADACONVF_H

#include "string.h"
#include"camada.h"
#include "funcoesDeAtivacao.h"

typedef struct CamadaConvF_t {
	Camada_t super;
	Tensor W;
	Tensor z;
	Tensor dz;
	Tensor dW;
	size_t passox, passoy;
	Kernel convFSum;
	Kernel convFCalcGradZ;
	Kernel convFCalcGradAndFixWeight;
	Kernel convFCalcGrads;
	Kernel convFCalcGradBatch;
	Kernel kernel_fixW;
	uint32_t activationFuntion;
	uint32_t derivationFuntion;
	RandomParams rdp_filtros;
} *CamadaConvF, CamadaConvF_t;

extern Camada CamadaConvF_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, uint32_t ativacao,
							  Tensor entrada, Parametros params, Ecx ecx, RandomParams rdp_filtros);
extern Camada CamadaConvF_load(FILE *f, Gpu gpu, Queue queue,  Tensor entrada, Ecx ecx);
#endif //CNN_GPU_CAMADACONVF_H
