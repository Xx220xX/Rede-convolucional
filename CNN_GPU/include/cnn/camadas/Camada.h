//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_H
#define CNN_GPU_CAMADA_H

#include "config.h"
#include"tensor/Tensor.h"
#include "utils/memory_utils.h"
#include "cnn_errors_list.h"


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
#define PRELU     11


typedef struct {
	double hitLearn, momento, decaimentoDePeso;
} Params;


typedef int (*f2v)(void *, void *);

typedef int (*fv)(void *);

typedef int (*fvc)(void *, char);

typedef int (*fv3d)(void *, double, double, double);

typedef int  (*f4v)(void *, void *, void *, void *);

typedef const char *(*cfv)(void *);

typedef struct {
	char type;
	char flag_releaseInput;
	char learnable;
	Params parametros;
	Tensor gradsEntrada;
	Tensor entrada;
	Tensor saida;
	QUEUE queue;
	cl_context context;
	size_t *max_works;
	f2v backpropagation;
	fv propagation;
	fv release;
	f4v salvar;
	cfv toString;
	cfv getCreateParams;
	fvc setLearn;
	fv3d setParams;
	char *__string__;
} *Camada, Typecamada;

void __newCamada__(Camada c, WrapperCL *cl, char type, Tensor entrada, QUEUE queue,
				   Params params, size_t xi, size_t yi, size_t zi,
				   size_t xo, size_t yo, size_t zo,
				   CNN_ERROR *error);

void __releaseCamada__(Camada c);

void CamadaSetLearn(Camada c, char learn);

void CamadaSetParams(Camada c, double hitlearn, double momento, double decaimento);

Camada carregarConv(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarPool(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarRelu(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarPRelu(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarDropOut(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarFullConnect(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarCamada(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarBatchNorm(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarSoftMax(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, CNN_ERROR *error);

Camada carregarPadding(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarPoolAv(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

Camada carregarConvNc(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error);

#endif //CNN_GPU_CAMADA_H
