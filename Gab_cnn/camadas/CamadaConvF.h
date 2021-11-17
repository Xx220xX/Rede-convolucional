
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
	Kernel kernelConvFSum;
	Kernel kernelConvFFixWeight;
	Kernel kernelConvFCalcZGrad;
	Kernel kernelConvFCalcGrads;
	int activationFuntion;
	int derivationFuntion;
} *CamadaConvF, CamadaConvF_t;

extern Camada createConvF(Gpu gpu, QUEUE queue, Ponto3d passo, Ponto3d filtro, Ponto3d size_in, int ativacao,
				   Tensor entrada, Params params, RandomParam randomParams);

#endif //CNN_GPU_CAMADACONV_H
