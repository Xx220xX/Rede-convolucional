//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_H
#define CNN_GPU_CAMADA_H

#include"../tensor/Tensor.h"
#include <time.h>

typedef struct {
	double hitLearn, momento, decaimentoDePeso;
} Params;
// funcao com dois parametros do tipo ponteiro com retorno inteiro

typedef int (*fvv)(void *, void *);

typedef int (*fv)(void *);

typedef void  (*fsl)(void *, void *, void *, void *);

typedef const char *(*fch)(void *);

#define CONV        1
#define POOL        2
#define RELU        3
#define DROPOUT     4
#define FULLCONNECT 5
#define SOFTMAX     6
#define BATCHNORM   7
#define PADDING     8
#define POOLAV      9
#define CONVNC     10


typedef struct {
	char type;
	Params *parametros;
	Tensor gradsEntrada;
	Tensor entrada;
	Tensor saida;
	char flag_releaseInput;
	char flag_notlearn;
	cl_command_queue queue;
	cl_context context;
	size_t *max_works;
	fvv calc_grads;
	fv corrige_pesos;
	fv ativa;
	fsl salvar;
	fch toString;
	char *__string__;
	fv release;
} *Camada, Typecamada;

void __newCamada__(Camada c, WrapperCL *cl, char type, Tensor entrada, cl_command_queue queue, Params *params, size_t xi,
              size_t yi, size_t zi, size_t xo, size_t yo, size_t zo, GPU_ERROR *error);

Camada carregarConv(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarPool(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarRelu(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error);

Camada
carregarDropOut(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error);

Camada
carregarFullConnect(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error);

Camada
carregarCamada(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *param, GPU_ERROR *error);

Camada
carregarBatchNorm(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarPadding(WrapperCL *cl, FILE *src, cl_command_queue queue,
                       Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarPoolAv(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error);

#endif //CNN_GPU_CAMADA_H
