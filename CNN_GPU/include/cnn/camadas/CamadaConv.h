
//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONV_H
#define CNN_GPU_CAMADACONV_H

#include "string.h"
#include"Camada.h"
#include <stdlib.h>
#include"utils.h"


typedef unsigned int UINT;


typedef struct {
	Typecamada super;
	Tensor filtros;
	Tensor grad_filtros;
	Tensor gradnext;
	UINT passox;
	UINT passoy;
	Kernel kernelConvSum;
	Kernel kernelConvFixWeight;
	Kernel kernelConvCalcGrads;
} *CamadaConv, Typecamadaconv;

Camada createConv(WrapperCL *cl, QUEUE queue, UINT passox, UINT passoy, UINT lenFilterx, UINT lenFiltery,
				  UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
				  Tensor entrada, Params params, CNN_ERROR *error, int randomize);

void releaseConv(CamadaConv *pc);



#endif //CNN_GPU_CAMADACONV_H
