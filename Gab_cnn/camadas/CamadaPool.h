//
// Created by Xx220xX on 25/10/2020.
//

#ifndef CNN_GPU_CAMADAPOOL_H
#define CNN_GPU_CAMADAPOOL_H

#include"camada.h"


typedef struct CamadaPool_t {
	Camada_t super;
	uint32_t type;
	size_t passox, passoy;
	size_t filtrox, filtroy;
	Kernel poolativa;
	Kernel poolCalcGrads;
} *CamadaPool, CamadaPool_t;

extern Camada CamadaPool_new(Gpu gpu, Queue queue, P2d passo, P2d filtro, P3d size_in, uint32_t type_pooling, Tensor entrada, Ecx ecx);


#endif //CNN_GPU_CAMADAPOOL_H
