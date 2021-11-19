//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADADROPOUT_H
#define CNN_GPU_CAMADADROPOUT_H

#include"camada.h"


typedef struct v {
	Camada_t super;
	Tensor hitmap;
	char flag_releaseInput;
	REAL p_ativacao;
	cl_ulong seed;
	Kernel kerneldropativa;
	Kernel kerneldropcalcgrad;
} *CamadaDropOut, CamadaDropOut_t;


extern Camada createDropOut(Gpu gpu, QUEUE queue, P3d size_in,
							REAL p_ativacao, long long int seed,
							Tensor entrada);


#endif //CNN_GPU_CAMADADROPOUT_H
