
//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONV_H
#define CNN_GPU_CAMADACONV_H

#include "string.h"
#include"Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>
#include"utils.h"


typedef unsigned int UINT;


typedef struct {
	Typecamada super;
	Tensor filtros;
	Tensor grad_filtros;
	Tensor grad_filtros_old;
	UINT passo, tamanhoFiltro, numeroFiltros;
	Kernel kernelConvSum;
	Kernel kernelConvFixWeight;
	Kernel kernelConvCalcGradsFiltro;
	Kernel kernelConvCalcGrads;
} *CamadaConv, Typecamadaconv;

void calc_gradsConv(CamadaConv c, Tensor Gradnext);

void releaseConv(CamadaConv *pc);

int ativaConv(CamadaConv c);

void corrige_pesosConv(CamadaConv c);

int convRandomize(CamadaConv c, WrapperCL *cl, GPU_ERROR *error);

void salvarConv(WrapperCL *cl, CamadaConv c, FILE *dst, GPU_ERROR *error);

Camada createConv(WrapperCL *cl, QUEUE queue, UINT passo, UINT lenFilter,
                  UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params params, GPU_ERROR *error, int randomize);


#endif //CNN_GPU_CAMADACONV_H
