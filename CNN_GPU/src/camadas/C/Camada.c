//
// Created by Henrique on 5/8/2021.
//

#include "../Camada.h"

Camada carregarCamada(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada,
                      Params param, Exception *error) {
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
			return carregarBatchNorm(cl, src, queue, entrada, param, error);
		case SOFTMAX:
			return carregarSoftMax(cl, src, queue, entrada, param, error);
		case POOLAV:
			return carregarPoolAv(cl, src, queue, entrada, param, error);
		case PADDING:
			return carregarPadding(cl, src, queue, entrada, param, error);
		case CONVNC:
			return carregarConvNc(cl, src, queue, entrada, param, error);
		default:
			return NULL;
	}

}

void __newCamada__(Camada c, WrapperCL *cl, char type, Tensor entrada, QUEUE queue,
                   Params params, size_t xi,
                   size_t yi, size_t zi, size_t xo, size_t yo, size_t zo, char usehost, Exception *error) {
	cl_context context = cl->context;
	if (error->error)return;
	c->flag_usehost = usehost;
	c->queue = queue;
	c->type = type;
	c->entrada = entrada;
	if (!entrada) {
		c->entrada = newTensor(context, c->queue, xi, yi, zi,c->flag_usehost, error);
		c->flag_releaseInput = 1;
	}
	c->saida = newTensor(context, queue, xo, yo, zo,c->flag_usehost, error);
	c->gradsEntrada = newTensor(context, queue, xi, yi, zi,c->flag_usehost, error);;
	c->parametros = params;
	c->max_works = &cl->maxworks;
	c->context = cl->context;

}

void __releaseCamada__(Camada c){
	if(c->entrada!=c->gradsEntrada){
		releaseTensor(&c->gradsEntrada);
	}
	if (c->flag_releaseInput){
		releaseTensor(&c->entrada);
	}
	releaseTensor(&c->saida);

}
void CamadaSetLearn(Camada c, char learn) {
	c->flag_notlearn = !learn;
}

void CamadaSetParams(Camada c, double hitlearn, double momento, double decaimento) {
	c->parametros.hitLearn = hitlearn;
	c->parametros.momento = momento;
	c->parametros.decaimentoDePeso = decaimento;
}