//
// Created by Xx220xX on 25/10/2020.
//

#ifndef CNN_GPU_CAMADAPOOL_H
#define CNN_GPU_CAMADAPOOL_H

#include "string.h"
#include"Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>
#include <float.h>

typedef unsigned int UINT;


typedef struct {
	Typecamada super;
	UINT passo;
	UINT tamanhoFiltro;

	Kernel kernelPoolAtiva;
	Kernel kernelPoolCalcGrads;
} *CamadaPool, Typecamadapool;

void releasePool(CamadaPool *pc);

void ativaPool(CamadaPool c);

void corrige_pesosPool(CamadaPool c);

void calc_gradsPool(CamadaPool c, Tensor GradNext);

void salvarPool(WrapperCL *cl, CamadaPool c, FILE *dst, GPU_ERROR *error);

Camada
createPool(WrapperCL *cl, cl_command_queue queue, UINT passo, UINT tamanhoFiltro, UINT inx, UINT iny, UINT inz,
           Tensor entrada, Params *params,
           GPU_ERROR *error);



#endif //CNN_GPU_CAMADAPOOL_H
