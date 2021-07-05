//
// Created by Henrique on 5/8/2021.
//

#include "Camada.h"

Camada
carregarCamada(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *param, GPU_ERROR *error) {
	char identify = 0;
	fread(&identify, sizeof(char), 1, src);
	if (feof(src))return NULL;
	switch (identify) {
		case CONV:
			return carregarConv(cl, src, queue, entrada, param, error);
		case POOL:
			return carregarPool(cl, src, queue, entrada, param, error);
		case RELU:
			return carregarRelu(cl, src, queue, entrada, param, error);
		case DROPOUT:
			return carregarDropOut(cl, src, queue, entrada, param, error);
		case FULLCONNECT:
			return carregarFullConnect(cl, src, queue, entrada, param, error);
		case BATCHNORM:
			return carregarBatchNorm(cl,src,queue,entrada,param,error);
		case POOLAV:
			return carregarPoolAv(cl,src,queue,entrada,param,error);
		case PADDING:
			return carregarPadding(cl,src,queue,entrada,param,error);
		case CONVNC:
			return carregarConvNc(cl,src,queue,entrada,param,error);
		default:
			return NULL;
	}

}

void __newCamada__(Camada c, WrapperCL *cl, char type, Tensor entrada, cl_command_queue queue, Params *params, size_t xi,
                   size_t yi, size_t zi, size_t xo, size_t yo, size_t zo, GPU_ERROR *error) {
	cl_context context = cl->context;
	if (error->error)return;
	c->type = type;
	c->entrada = entrada;
	if (!entrada) {
		c->entrada = newTensor(context, xi, yi, zi, error);
		c->flag_releaseInput = 1;
	}

	c->saida = newTensor(context, xo, yo, zo, error);
	c->gradsEntrada = newTensor(context, xi, yi, zi, error);
	c->parametros = params;
	c->max_works = &cl->maxworks;
	c->context = cl->context;
	c->queue = queue;
}

