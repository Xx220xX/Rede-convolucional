
//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONVF_H
#define CNN_GPU_CAMADACONVF_H

#include "string.h"
#include"Camada.h"
#include <stdlib.h>
#include"utils.h"
#include "funcoesDeAtivacao.h"

typedef unsigned int UINT;


typedef struct {
	Typecamada super;
	Tensor filtros;
	Tensor z;
	Tensor dz;
	Tensor grad_filtros;
	UINT passox;
	UINT passoy;
	Kernel kernelConvFSum;
	Kernel kernelConvFFixWeight;
	Kernel kernelConvFCalcZGrad;
	Kernel kernelConvFCalcGrads;
	int activationFuntion;
	int derivationFuntion;
} *CamadaConvF, Typecamadaconvf;

Camada createConvF(WrapperCL *cl, QUEUE queue, UINT passox, UINT passoy, UINT lenFilterx, UINT lenFiltery,
				  UINT numeroFiltros, UINT inx, UINT iny, UINT inz,int ativacao,
				  Tensor entrada, Params params, RandomParam randomParams, CNN_ERROR *error);

void releaseConvF(CamadaConvF *pc);


#endif //CNN_GPU_CAMADACONV_H
