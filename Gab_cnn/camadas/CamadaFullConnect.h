//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADAFULLCONNECT_H
#define CNN_GPU_CAMADAFULLCONNECT_H

#include"camada.h"

#include"funcoesDeAtivacao.h"


typedef struct CamadaFullConnect_t{
	Camada_t super;
	Tensor pesos;
	Tensor grad;
	Tensor z;
	Tensor dz;
	// funcao de ativacao e sua derivada
	int fa, dfa;
	Kernel kernelfullfeed;
	Kernel kernelfullfixWeight;
	Kernel kernelfullcalcgrad1;
	Kernel kernelfullcalcgrad2;
} *CamadaFullConnect, CamadaFullConnect_t;

extern Camada createFullConnect(Gpu gpu, QUEUE queue, P3d size_in, size_t tamanhoSaida,
								Tensor entrada, Params params, int funcaoDeAtivacao, RandomParam randomParams);


#endif //CNN_GPU_CAMADAFULLCONNECT_H
