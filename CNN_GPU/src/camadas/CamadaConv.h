
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

Camada createConv(WrapperCL *cl, QUEUE queue, UINT passo, UINT lenFilter,
                  UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params params, char usehost, Exception *error, int randomize);

void releaseConv(CamadaConv *pc);



#endif //CNN_GPU_CAMADACONV_H
