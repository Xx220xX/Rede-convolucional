//
// Created by Henrique on 8/5/2021.
//

#ifndef CNN_GPU_CAMADABATCHNORM_H
#define CNN_GPU_CAMADABATCHNORM_H

#include "camada.h"
#include <stdlib.h>


typedef struct CamadaBatchNorm_t {
	/// classe mãe Camada_t
	Camada_t super;
	/// batch atual
	size_t batch;
	/// tamanho do lote
	size_t batch_size;
	/// Gama
	Tensor Y;
	/// gradiente de Gama
	Tensor dY;
	/// BETA
	Tensor B;
	/// grandiente de BETA
	Tensor dB;
	/// armazena a media do tensor de entrada
	Tensor media;
	/// armazena o inverso do desvio padrão
	Tensor inv_std;

	/// armazena a entrada normalizada
	Tensor norma;//(x,y,z)
	Tensor dnorma;//(x,y,z)
	Tensor media_dnorma;//(1,1,z)
	Tensor media_dnorma_norma;//(1,1,z)

	/// constante para evitar divisão por zero
	REAL epsilon
	;
	Kernel BatchNormMedia;// calcula a media
	Kernel BatchNormInvDesv;// calcula a variancia
	Kernel BatchNormNormaliza;// normaliza

	Kernel BatchNormaCalcDnorm;// calcula gradientes da norma
	Kernel BatchNormMediadnorm_norma;// calcula a media da dnorma e a media da dnorma .* norma
	Kernel BatchNormaCalcDa;// calcula gradientes de entrada

	Kernel BatchNormaCalcdYdB;// calcula gradientes de entrada
	Kernel BatchNormaLearn;// calcula gradientes de entrada
	/// Parametros de geração aleatória do tensor Gama
	RandomParams rdp_Y;
	/// Parametros de geração aleatória do tensor Beta
	RandomParams rdp_B;
} *CamadaBatchNorm, CamadaBatchNorm_t;

extern Camada CamadaBatchNorm_new(INTERNAL_DEFAULT_ARGS, Parametros params, REAL epsilon, size_t batchSize, Rdp randY, Rdp randB);

extern Camada CamadaBatchNorm_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADABATCHNORM_H
