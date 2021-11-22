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

#define CONVOLUCAO_ID 	1
#define CONVOLUCAOF_ID 	2
#define CONVOLUCAONC_ID 3
#define POOLING_ID 		4
#define FULLCONNECT_ID 	5
#define PADDING_ID 		6
#define DROPOUT_ID 		7
#define RELU_ID 		8
#define PRELU_ID 		9
#define SOFTMAX_ID 		9

typedef struct Camada_t {
	/// nome canonico da camada (apenas leitura)
	const char *layer_name;
	/// identificador da camada (apenas leitura)
	const int layer_id;
	/// parametros da camada
	Parametros params;
	/// entrada
	Tensor a;
	/// gradiente de entrada
	Tensor da;
	/// saída
	Tensor s;
	/// fila para utilizar gpu
	void *queue;
	/// numero maximo de threads na gpu
	size_t *maxcompute;
	/// variavel de controle de erro
	Ecx erro;
	/// tamanho da entrada
	P3d size_in;

	/// faz a propagação na camada, o Tensro de entrada é usado a
	int (*propagation)(void *self);
	/// faz a retropropagação, ds deve ser o gradiente da saída
	int (*retroPropagation)(void *self, Tensor ds);
	/// retorna uma string (que deve ser liberada com free_mem) contendo o objeto no formato json
	char *(*json)(void *self, int showTensorValues);
	/// retorna a chamada do construtor que gerou essa camada
	char *(*getGenerate)(void *self);
	/// libera os recursos alocados pela camada
	void (*release)(void *self_p);
	/// salva a camada no arquivo destino (apenas pesos são salvos, os gradientes são descartados)
	int (*save)(void *self, FILE *destino);
	/// retorna o tamanho da saída dessa camada
	P3d (*getOutSize)(void *self);
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

void internal_Camada_new(Camada self, Gpu gpu, Queue queue, int layer_id, const char *layer_name, Parametros params,
						 Tensor entrada, P3d dim_in, P3d dim_out, Ecx erro);

void internal_Camada_release(Camada *self);

char *internal_json(Camada self, int showValues);

#define Execute(kernel,len,...)if(!self->super.erro->error)self->super.erro->setError(self->super.erro, \
self->kernel->runRecursive(self->kernel, self->super.queue,len,*self->super.maxcompute, ##__VA_ARGS__))
#define Release(self)if(self)(self)->release(&(self))
#define CheckKernel(kernel)if (self->super.erro->setError(self->super.erro, self->kernel->error))goto methods
#define apendTensor(name,t,string, len,tmp,showValues)\
tmp = self->t->json(self->t, showValues);\
apendstr(string, len, ",\n"PAD"\""name"\":%s", tmp);\
free_mem(tmp)
#endif //GAB_CNN_CAMADA_H
