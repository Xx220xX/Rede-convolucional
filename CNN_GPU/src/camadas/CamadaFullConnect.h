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
	Tensor dz;
	Tensor dz_old;
	// funcao de ativacao e sua derivada
	int fa, dfa;
	Kernel kernelfullfeed;
	Kernel kernelfullfixWeight;
	Kernel kernelfullcalcgrad1;
	Kernel kernelfullcalcgrad2;
} *CamadaFullConnect, Typecamadafullconnect;

void releaseFullConnect(CamadaFullConnect *pc);


void corrigePesosFullConnect(CamadaFullConnect c);

void ativaFullConnect(CamadaFullConnect c);

void calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext);

int fullRandomize(CamadaFullConnect c, WrapperCL *cl, GPU_ERROR *error);

void salvarFullConnect(WrapperCL *cl, CamadaFullConnect c, FILE *dst, GPU_ERROR *error);

Camada createFullConnect(WrapperCL *cl, QUEUE queue, UINT inx, UINT iny, UINT inz, UINT tamanhoSaida,
                         Tensor entrada, Params params,
                         int funcaoDeAtivacao, int randomize, GPU_ERROR *error);



#endif //CNN_GPU_CAMADAFULLCONNECT_H
