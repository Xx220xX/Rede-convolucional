//
// Created by Henrique on 8/5/2021.
//

#ifndef CNN_GPU_CAMADABATCHNORM_H
#define CNN_GPU_CAMADABATCHNORM_H

#include "camada.h"
#include <stdlib.h>

#define UINT unsigned int;
#define ULL unsigned long long int

typedef struct CamadaBatchNorm_t {
	Camada_t super;
	Tensor Y;
	Tensor gradY;
	Tensor B;
	Tensor gradB;
	Tensor media;
	Tensor somaDiferenca;
	Tensor variancia;
	Tensor gradVariancia;
	Tensor diferenca;
	Tensor diferencaquad;
	Tensor norma;

	REAL epsilon;

	Kernel batchNormAtiva1;// calcula a media
	Kernel batchNormAtiva2;// calcula a diferenca
	Kernel batchNormAtiva3;// calcula a variancia
	Kernel batchNormAtiva4;// normaliza
	Kernel batchNormCalcGrads1;// calcula gradientes de entrada
	Kernel batchNormCalcGrads2;//calcula gradiente Y B

	RdP rdp_Y;
	RdP rdp_B;
} *CamadaBatchNorm, CamadaBatchNorm_t;

extern Camada CamadaBatchNorm_new(Gpu gpu, Queue queue, Parametros params, P3d size_in, Tensor entrada, REAL epsilon,Ecx ecx, RdP randY, RdP randB);


#endif //CNN_GPU_CAMADABATCHNORM_H
