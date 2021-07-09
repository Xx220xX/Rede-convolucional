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


Camada createPool(WrapperCL *cl, QUEUE queue, UINT passo, UINT tamanhoFiltro,
                  UINT inx, UINT iny, UINT inz, Tensor entrada, Params params,
                 char usehost, GPU_ERROR *error);


#endif //CNN_GPU_CAMADAPOOL_H
