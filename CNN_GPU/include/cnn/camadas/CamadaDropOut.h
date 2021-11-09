//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADADROPOUT_H
#define CNN_GPU_CAMADADROPOUT_H

#include "string.h"
#include"Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>
#include <float.h>


typedef struct {
	Typecamada super;
	Tensor hitmap;
	char flag_releaseInput;
	REAL p_ativacao;
	cl_ulong seed;
	Kernel kerneldropativa;
	Kernel kerneldropcalcgrad;
} *CamadaDropOut, Typecamadadropout;


Camada createDropOut(WrapperCL *cl, QUEUE queue, UINT inx, UINT iny, UINT inz,
					 REAL p_ativacao, long long seed,
					 Tensor entrada,  CNN_ERROR *error);




#endif //CNN_GPU_CAMADADROPOUT_H
