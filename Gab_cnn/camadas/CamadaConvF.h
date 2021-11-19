
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
	Tensor filtros;
	Tensor z;
	Tensor dz;
	Tensor grad_filtros;
	size_t passox, passoy;
	Kernel convFSum;
	Kernel convFCalcGradZ;
	Kernel convFCalcGradAndFixWeight;
	Kernel convFCalcGrads;
	int activationFuntion;
	int derivationFuntion;
	RdP rdp_filtros;
} *CamadaConvF, CamadaConvF_t;

extern Camada CamadaConvF_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, int ativacao,
							  Tensor entrada, Parametros params, Ecx ecx, RdP rdp_filtros);

#endif //CNN_GPU_CAMADACONVF_H
