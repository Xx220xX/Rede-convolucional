//
// Created by Henrique on 8/5/2021.
//

#ifndef CNN_GPU_CAMADABATCHNORM_H
#define CNN_GPU_CAMADABATCHNORM_H

#include "camada.h"
#include <stdlib.h>

#define UINT unsigned int;
#define ULL unsigned long long int

typedef struct CamadaBatchNorm_t{
	Camada_t super;
	Tensor Y;
	Tensor gradY;
	Tensor B;
	Tensor gradB;

	REAL epsilon;


	Tensor media;

	Tensor somaDiferenca;
	Tensor variancia;
	Tensor gradVariancia;
	Tensor diferenca;


	Tensor diferencaquad;
	Tensor norma;

	Kernel kernelBatchNormAtiva1;// calcula a media
	Kernel kernelBatchNormAtiva2;// calcula a diferenca
	Kernel kernelBatchNormAtiva3;// calcula a variancia
	Kernel kernelBatchNormAtiva4;// normaliza
	Kernel kernelBatchNormCalcGrads1;// calcula gradientes de entrada
	Kernel kernelBatchNormCalcGrads2;//calcula gradiente Y B
	Kernel kernelBatchNormCorrige;//arruma os coeficientes

} *CamadaBatchNorm, CamadaBatchNorm_t;

extern Camada createBatchNorm(Gpu gpu, QUEUE queue, Params params, Ponto3d size_in,Tensor entrada, REAL epsilon, RandomParam randY, RandomParam randB);


#endif //CNN_GPU_CAMADABATCHNORM_H
