//
// Created by Henrique on 8/5/2021.
//

#ifndef CNN_GPU_CAMADABATCHNORM_H
#define CNN_GPU_CAMADABATCHNORM_H

#include "Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>

typedef unsigned int UINT;
#define ULL unsigned long long int
typedef struct {
	Typecamada super;
	Tensor Y;
	Tensor gradY;
	Tensor B;
	Tensor gradB;

	double epsilon;


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

} *CamadaBatchNorm, TypecamadaBatchNorm;


Camada createBatchNorm(WrapperCL *cl, QUEUE queue, Params params, unsigned int inx, unsigned int iny,
					   unsigned int inz, Tensor entrada, double epsilon, RandomParam randY, RandomParam randB, CNN_ERROR *error);


#endif //CNN_GPU_CAMADABATCHNORM_H
