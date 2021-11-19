//
// Created by Xx220xX on 25/10/2020.
//

#ifndef CNN_GPU_CAMADAPOOL_H
#define CNN_GPU_CAMADAPOOL_H

#include"camada.h"

#define MAXPOOL 1
#define MINPOOL 2
#define AVEPOOL 3
typedef struct CamadaPool_t {
	Camada_t super;
	int type;
	size_t passox, passoy;
	size_t filtrox, filtroy;
	Kernel poolativa;
	Kernel poolCalcGrads;
} *CamadaPool, CamadaPool_t;

extern Camada CamadaPooling_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, int type_pooling, Tensor entrada, Ecx ecx);


#endif //CNN_GPU_CAMADAPOOL_H
