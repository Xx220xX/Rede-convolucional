//
// Created by hslhe on 13/11/2021.
//

#ifndef GAB_CNN_CAMADA_H
#define GAB_CNN_CAMADA_H

#include <gpu/Gpu.h>
#include <gpu/Kernel.h>
#include "tensor/tensor.h"
#include "parametros.h"
#include "ponto3d.h"

#define CONVOLUCAO_ID 1

#define UINT unsigned int
typedef struct Camada_t {
	const char *layer_name;
	const int layer_id;
	Parametros params;
	Tensor a;
	Tensor da;
	Tensor s;
	void *queue;
	char release_da;
	size_t *maxcompute;
	Ecx erro;
	Ponto3d size_in;

	int (*propagation)(void *self);

	int (*retroPropagation)(void *self, Tensor *ds);

	char *(*json)(void *self, int showTensorValues);

	char *(*getGenerate)(void *self);

	void (*release)(void *self_p);

	Ponto3d (*getOutSize)(void *self);
} *Camada, Camada_t;

/**
 * Parametros para função aleatoria
 *  Y = X * a + b
 *  @param type: indica o tipo da distribuicao pode ser TENSOR_NORMAL para gaussiana, TENSOR_UNIFORM para uniforme
 *  0 para iniciar por padrão
 *  -1 para nao aleatoriazar o tensor
 */
typedef struct RdParams {
	REAL a, b;
	int type;
} RdP;

void internal_Camada_new(Camada self, Gpu gpu, Queue queue,  int layer_id, const char *layer_name, Parametros params,
						 Tensor entrada, Ponto3d dim_in, Ponto3d dim_out, Ecx erro);

void internal_Camada_release(Camada *self);

char *internal_json(Camada self, int showValues);

#endif //GAB_CNN_CAMADA_H
