//
// Created by Xx220xX on 25/10/2020.
//

#ifndef CNN_GPU_CAMADAPOOLAV_H
#define CNN_GPU_CAMADAPOOLAV_H

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
	Kernel kernelPoolAvAtiva;
	Kernel kernelPoolAvCalcGrads;
} *CamadaPoolAv, TypecamadaPoolAv;

void releasePoolAv(CamadaPoolAv *pc);

void ativaPoolAv(CamadaPoolAv c);

void corrige_pesosPoolAv(CamadaPoolAv c);

void calc_gradsPoolAv(CamadaPoolAv c, Tensor GradNext);

void salvarPoolAv(WrapperCL *cl, CamadaPoolAv c, FILE *dst, GPU_ERROR *error);

Camada createPoolAv(WrapperCL *cl, cl_command_queue queue, UINT passo, UINT tamanhoFiltro, UINT inx, UINT iny, UINT inz,
           Tensor entrada, Params *params,
           GPU_ERROR *error);



#endif //CNN_GPU_CAMADAPoolAv_H
