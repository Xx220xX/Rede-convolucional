//
// Created by gf154 on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_PRELU_H
#define CNN_GPU_CAMADA_PRELU_H

#include "camada.h"


typedef struct CamadaPRelu_t {
	Camada_t super;
	Tensor A;
	Tensor dA;
	RandomParams rdp_a;
	Kernel preluativa;
	Kernel preluonlyfix;
	Kernel prelucalcgrad;
} *CamadaPRelu, CamadaPRelu_t;

extern Camada CamadaPRelu_new(Gpu gpu, Queue queue, P3d size_in, Tensor entrada, Parametros params, RandomParams rdp_a, Ecx ecx);
extern Camada CamadaPRelu_load(FILE *f, Gpu gpu, Queue queue, P3d sizein, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADA_PRelu_H
