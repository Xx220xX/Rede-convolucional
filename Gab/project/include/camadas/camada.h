//
// Created by hslhe on 13/11/2021.
//

#ifndef GAB_CNN_CAMADA_H
#define GAB_CNN_CAMADA_H

#include <gpu/Gpu.h>
#include <gpu/Kernel.h>
#include "tensor/tensor.h"
#include "cnn/parametros.h"
#include "cnn/ponto3d.h"
#include "funcoesDeAtivacao.h"
#include "error_list.h"

#define CONVOLUCAO_ID         0x1
#define CONVOLUCAOF_ID         0x2
#define CONVOLUCAONC_ID         0x3
#define POOL_ID                     0x4
#define FULLCONNECT_ID             0x5
#define PADDING_ID                 0x6
#define DROPOUT_ID             0x7
#define RELU_ID                 0x8
#define PRELU_ID                 0x9
#define SOFTMAX_ID             0xA
#define BATCHNORM_ID             0xB

/// implementa a maxpooling
#define MAXPOOL                 0x1
/// implementa a minpooling
#define MINPOOL                 0x2
/// implementa a averagepooling
#define AVEPOOL                 0x3

/// Indica que a softmax é a ultima camada .
#define SOFTLAST                 0x1
/// Indica que a softmax vai subtrair o maximo da entrada .
#define SOFTNORM                 0x2
/// Indica que a softmax não é a ultima camada (default) .
#define SOFTNLAST                 0x0
/// Indica que a softmax não vai subtrair o maximo da entrada (default) .
#define SOFTNNORM                 0x0


typedef struct Camada_t {
	/// nome canonico da camada (apenas leitura)
	const char *layer_name;
	/// identificador da camada (apenas leitura)
	const char layer_id;
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
	Ecx ecx;
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
	int type;
	REAL a, b;
} RandomParams, RdParams;

#define  RDP(type, ...)((RandomParams){type,## __VA_ARGS__})

void internal_Camada_new(Camada self, Gpu gpu, Queue queue, char layer_id, const char *layer_name, Parametros params, Tensor entrada, P3d dim_in, P3d dim_out, Ecx erro);

void internal_Camada_release(Camada *self);

char *internal_json(Camada self, int showValues);

void internal_saveCamada(FILE *f, Camada self);

void internal_loadCamada(FILE *f, Parametros *parametros, P3d *size_in, uint32_t *size_element);

void internal_saveTensor(FILE *f, Tensor t);

void internal_loadTensor(FILE *f, Tensor t, uint32_t size_element);

void internal_saveREAL(FILE *f, REAL value);

void internal_loadREAL(FILE *f, REAL *value, uint32_t size_element);

RdParams internal_getDefaultRDP(int is_reluActivation, size_t inputLength, size_t outLength);

#define Execute(kernel, len, ...)if(!self->super.ecx->error)self->super.ecx->setError(self->super.ecx, \
self->kernel->runRecursive(self->kernel, self->super.queue,len,*self->super.maxcompute, ##__VA_ARGS__))
#define Release(self)if(self)(self)->release(&(self));(self)=NULL
#define CheckKernel(kernel)if (self->super.ecx->setError(self->super.ecx, self->kernel->error))goto methods
#define apendTensor(name, t, string, len, tmp, showValues) \
if(self->t)  {                                                     \
tmp = self->t->json(self->t, showValues);\
apendstr(string, len, ",\n"PAD"\""name"\":%s", tmp);\
free_mem(tmp);}\
else apendstr(string, len, ",\n"PAD"\""name"\": null")
#endif //GAB_CNN_CAMADA_H
