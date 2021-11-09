//
// Created by Henrique on 5/8/2021.
//

#include "camadas/Camada.h"
Camada carregarCamada(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada, CNN_ERROR *error) {
	char identify = 0;
	fread(&identify, sizeof(char), 1, src);
	if (feof(src))return NULL;
	switch (identify) {
		case CONV:
			return carregarConv(cl, src, queue, entrada,  error);
		case POOL:
			return carregarPool(cl, src, queue, entrada,  error);
		case RELU:
			return carregarRelu(cl, src, queue, entrada,  error);
		case PRELU:
			return carregarPRelu(cl, src, queue, entrada,  error);
		case DROPOUT:
			return carregarDropOut(cl, src, queue, entrada,  error);
		case FULLCONNECT:
			return carregarFullConnect(cl, src, queue, entrada,  error);
		case BATCHNORM:
			return carregarBatchNorm(cl, src, queue, entrada,  error);
		case SOFTMAX:
			return carregarSoftMax(cl, src, queue, entrada,  error);
		case POOLAV:
			return carregarPoolAv(cl, src, queue, entrada,  error);
		case PADDING:
			return carregarPadding(cl, src, queue, entrada,  error);
		case CONVNC:
			return carregarConvNc(cl, src, queue, entrada,  error);
		default:
			return NULL;
	}

}

void __newCamada__(Camada c, WrapperCL *cl, char type, Tensor entrada, QUEUE queue,
				   Params params, size_t xi,
				   size_t yi, size_t zi, size_t xo, size_t yo, size_t zo, CNN_ERROR *error) {
	cl_context context = cl->context;
	if (error->error)return;
	c->queue = queue;
	c->type = type;
	c->entrada = entrada;
	c->gradsEntrada = NULL;
	if (!entrada) {
		c->entrada = newTensor(context, c->queue, xi, yi, zi, 0, error);
		c->flag_releaseInput = 1;
	} else {
 		c->gradsEntrada = newTensor(context, queue, xi, yi, zi, 0, error);
	}
	c->saida = newTensor(context, queue, xo, yo, zo, 0, error);
	c->parametros = params;
	c->max_works = &cl->maxworks;
	c->context = cl->context;
	c->setLearn = (fvc) CamadaSetLearn;
	c->setParams = (fv3d) CamadaSetParams;
	c->learnable = 1;
}

void __releaseCamada__(Camada c) {
	releaseTensor(&c->gradsEntrada);
	releaseTensor(&c->saida);
	if (c->flag_releaseInput)releaseTensor(&c->entrada);
	if (c->__string__)free_mem(c->__string__);
	c->__string__ = NULL;
}

void CamadaSetLearn(Camada c, char learn) {
	c->learnable = learn != 0;
}

void CamadaSetParams(Camada c, REAL hitlearn, REAL momento, REAL decaimento) {
	c->parametros.hitLearn = hitlearn;
	c->parametros.momento = momento;
	c->parametros.decaimentoDePeso = decaimento;
}