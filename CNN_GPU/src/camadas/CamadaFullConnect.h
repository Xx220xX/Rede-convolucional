//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADAFULLCONNECT_H
#define CNN_GPU_CAMADAFULLCONNECT_H

#include "string.h"
#include"Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>
#include <float.h>

#include"funcoesDeAtivacao.h"


typedef struct {
	Typecamada super;
	Tensor z;
	Tensor pesos;
	Tensor grad;
	Tensor dz;
	// funcao de ativacao e sua derivada
	int fa, dfa;
	Kernel kernelfullfeed;
	Kernel kernelfullfixWeight;
	Kernel kernelfullcalcgrad1;
	Kernel kernelfullcalcgrad2;
} *CamadaFullConnect, Typecamadafullconnect;

Camada createFullConnect(WrapperCL *cl, QUEUE queue, UINT inx, UINT iny, UINT inz, UINT tamanhoSaida,
                         Tensor entrada, Params params, int funcaoDeAtivacao, int randomize,
                         char usehost, Exception *error);



#endif //CNN_GPU_CAMADAFULLCONNECT_H
