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
	Kernel kernelPoolAtiva;
	Kernel kernelPoolCalcGrads;
} *CamadaPool, CamadaPool_t;

extern Camada createPool(Gpu gpu, Queue queue, Ponto3d passo, Ponto3d filtro, Ponto3d size_in,int type_pooling, Tensor entrada);


#endif //CNN_GPU_CAMADAPOOL_H
