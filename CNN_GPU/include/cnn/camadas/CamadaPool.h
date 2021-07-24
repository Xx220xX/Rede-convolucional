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
	UINT passox;
	UINT passoy;
	UINT filtrox;
	UINT filtroy;

	Kernel kernelPoolAtiva;
	Kernel kernelPoolCalcGrads;
} *CamadaPool, Typecamadapool;

Camada createPool(WrapperCL *cl, cl_command_queue queue,
                  UINT passox, UINT passoy,
                  UINT filtrox, UINT filtroy,
                  UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params params,
                  char usehost, Exception *error);


#endif //CNN_GPU_CAMADAPOOL_H
