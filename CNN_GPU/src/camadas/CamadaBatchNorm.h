//
// Created by Henrique on 8/5/2021.
//

#ifndef CNN_GPU_CAMADABATCHNORM_H
#define CNN_GPU_CAMADABATCHNORM_H

#include "Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>

#define UINT unsigned int
#define ULL unsigned long long int
typedef struct {
	Typecamada super;
	Kernel kernelBatchNormAtiva1;// calcula a media
	Kernel kernelBatchNormAtiva2;// calcula a diferenca
	Kernel kernelBatchNormAtiva3;// calcula a variancia
	Kernel kernelBatchNormAtiva4;// normaliza
	Kernel kernelBatchNormCalcGrads1;// calcula gradientes de entrada
	Kernel kernelBatchNormCalcGrads2;//calcula gradiente Y B
	Kernel kernelBatchNormCorrige;//arruma os coeficientes


	double epsilon;

	Tensor media;
	Tensor somaDiferenca;
	Tensor variancia;
	Tensor gradVariancia;

	Tensor Y;
	Tensor B;
	Tensor gradY;
	Tensor gradB;
	Tensor diferenca;
	Tensor diferencaquad;

	Tensor norma;

} *CamadaBatchNorm, TypecamadaBatchNorm;

void realeaseBatchNorm(CamadaBatchNorm *pc);

void ativaBatchNorm(CamadaBatchNorm c);

void corrige_pesosBatchNorm(CamadaBatchNorm c);

void calc_gradsBatchNorm(CamadaBatchNorm c, Tensor GradNext);

void salvarBatchNorm(WrapperCL *cl, CamadaBatchNorm c, FILE *dst, GPU_ERROR *error);

Camada createBatchNorm(WrapperCL *cl, cl_command_queue queue, Params *params, unsigned int inx, unsigned int iny,
                       unsigned int inz, Tensor entrada, double epsilon, int randomize, GPU_ERROR *error);


#endif //CNN_GPU_CAMADABATCHNORM_H
