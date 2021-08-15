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
	UINT passox;
	UINT passoy;
	UINT fx;
	UINT fy;
	Kernel kernelPoolAvAtiva;
	Kernel kernelPoolAvCalcGrads;
} *CamadaPoolAv, TypecamadaPoolAv;


Camada createPoolAv(WrapperCL *cl, QUEUE queue, UINT px, UINT py, UINT fx, UINT fy,
					UINT inx, UINT iny, UINT inz,
					Tensor entrada, Params params,
					char usehost, CNN_ERROR *error);



#endif //CNN_GPU_CAMADAPoolAv_H
