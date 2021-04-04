#ifndef CNN_GPU_CNN_H
#define CNN_GPU_CNN_H

#include "Camada.h"
#include "CamadaConv.h"
#include "CamadaRelu.h"
#include "CamadaDropOut.h"
#include "CamadaFullConnect.h"
#include "CamadaPool.h"

#ifdef LOG_CNN_ADD_LAYERS
#undef LOG_CNN_ADD_LAYERS
#define LOG_CNN_ADD_LAYERS(format, ...) printf(format,## __VA_ARGS__);printf("\n");
#else
#define LOG_CNN_ADD_LAYERS(format, ...)
#endif


#define INVALID_FILTER_SIZE (-1)
#define CNN_FLAG_CALCULE_ERROR 1

typedef struct {
	Params parametros;
	Camada *camadas;
	int size;
	Ponto3d sizeIn;
	char err;
	cl_command_queue queue;
	WrapperCL *cl;
	char releaseCL;
	GPU_ERROR error;
	Kernel kernelsub;
	Kernel kerneldiv;
	Kernel kerneldivInt;
	Kernel kernelNorm;
	Kernel kernelNormalize;
	Kernel kernelfindExtreme;
	Kernel kernelMax;
	Kernel kernelInt2Vector;
	char flags;
	double normaErro;

} *Cnn, TypeCnn;

Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz) {
	Cnn c = (Cnn) calloc(1, sizeof(TypeCnn));
	snprintf(c->error.msg, 255, "");
	c->parametros = p;
	c->sizeIn = (Ponto3d) {inx, iny, inz};
	c->cl = cl;
	int error = 0;
	c->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error);
	if (error) {
		c->error.error = error;
		snprintf(c->error.msg, 255, "nao foi possivel criar queue\n");
	}
	c->kernelsub = new_Kernel(cl->program, "sub", 4, VOID_P, VOID_P, VOID_P, INT);
	c->kerneldiv = new_Kernel(cl->program, "div", 3, VOID_P, DOUBLE, INT);
	c->kerneldivInt = new_Kernel(cl->program, "divIntDo", 4, VOID_P, VOID_P, DOUBLE, INT);
	c->kernelNorm = new_Kernel(cl->program, "norm", 3, VOID_P, VOID_P, INT);
	c->kernelMax = new_Kernel(cl->program, "maxID", 3, VOID_P, VOID_P, INT);
	c->kernelInt2Vector = new_Kernel(cl->program, "int2vector", 4, VOID_P, VOID_P, INT, INT);

	c->kernelNormalize = new_Kernel(cl->program, "normalizeVector", 6, VOID_P, VOID_P, DOUBLE, DOUBLE, DOUBLE, INT);
	c->kernelfindExtreme = new_Kernel(cl->program, "findExtremes", 3, VOID_P, VOID_P, INT);
	setmaxWorks(cl->maxworks);
	return c;
}

void releaseCnn(Cnn *pc) {
	Cnn c = *pc;
	if (!c)return;
	for (int i = 0; i < c->size; ++i) {
		c->camadas[i]->release(c->camadas + i);
	}
	free(c->camadas);
	clReleaseCommandQueue(c->queue);
	Kernel_release(&c->kernelsub);
	Kernel_release(&c->kerneldiv);
	Kernel_release(&c->kerneldivInt);
	Kernel_release(&c->kernelNorm);
	Kernel_release(&c->kernelNormalize);
	Kernel_release(&c->kernelfindExtreme);
	Kernel_release(&c->kernelMax);
	if (c->releaseCL) {
		WrapperCL_release(c->cl);
		free(c->cl);
	}
	free(c);
	*pc = NULL;
}

Cnn createCnnWithgpu(char *kernelFile, Params p, UINT inx, UINT iny, UINT inz) {
	WrapperCL *cl = (WrapperCL *) calloc(sizeof(WrapperCL), 1);
	WrapperCL_initbyFile(cl, kernelFile);
	Cnn c = createCnn(cl, p, inx, iny, inz);
	c->releaseCL = 1;
	return c;
}

Ponto3d __addLayer(Cnn c) {
	c->size += 1;
	c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
	Ponto3d in = c->sizeIn;
	if (c->size > 1) {
		in.x = (int) c->camadas[c->size - 2]->saida->x;
		in.y = (int) c->camadas[c->size - 2]->saida->y;
		in.z = (int) c->camadas[c->size - 2]->saida->z;
	}
	return in;
}

#define checkSizeFilter(v, tam, pas) ((((v)-(tam))/(pas)) ==((double)(v)-(tam))/((double)(pas)))

int CnnAddConvLayer(Cnn c, UINT passo, UINT tamanhoDoFiltro, UINT numeroDeFiltros) {
	/** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
	* em que:
	* 			S = tamanho da saida
	* 			E = tamanho da entrada
	* 			F = tamanho do filtro
	* 			Pd = preenchimento com zeros
	* 			P = passo
	**/

	Ponto3d sizeIn = __addLayer(c);
	if (!checkSizeFilter(sizeIn.x, tamanhoDoFiltro, passo) || !checkSizeFilter(sizeIn.y, tamanhoDoFiltro, passo)) {
		c->err = INVALID_FILTER_SIZE;
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		fprintf(stderr, "tamanho do filtro invalido\n");
		return c->err;

	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createConv(c->cl, passo, tamanhoDoFiltro, numeroDeFiltros, sizeIn.x, sizeIn.y, sizeIn.z,
	                                     entrada,
	                                     &c->parametros, &c->error, 1);
	c->camadas[c->size - 1]->queue = c->queue;
	if (!c->error.error) {
		LOG_CNN_ADD_LAYERS("camada convolutiva adicionada");
		LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                   c->camadas[c->size - 1]->saida->z);
	}
	return c->error.error;
}

int CnnAddPoolLayer(Cnn c, UINT passo, UINT tamanhoDoFiltro) {
	/** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
	* em que:
	* 			S = tamanho da saida
	* 			E = tamanho da entrada
	* 			F = tamanho do filtro
	* 			Pd = preenchimento com zeros
	* 			P = passo
	**/

	Ponto3d sizeIn = __addLayer(c);
	if (!checkSizeFilter(sizeIn.x, tamanhoDoFiltro, passo) || !checkSizeFilter(sizeIn.y, tamanhoDoFiltro, passo)) {
		c->err = INVALID_FILTER_SIZE;
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->err;

	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createPool(c->cl, passo, tamanhoDoFiltro, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
	                                     &c->parametros, &c->error);
	c->camadas[c->size - 1]->queue = c->queue;

	LOG_CNN_ADD_LAYERS("camada pooling adicionada");
	LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
	                   c->camadas[c->size - 1]->saida->z,);

	return c->error.error;

}

int CnnAddReluLayer(Cnn c) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createRelu(c->cl, sizeIn.x, sizeIn.y, sizeIn.z, entrada, &c->error);
	c->camadas[c->size - 1]->queue = c->queue;
	if (!c->error.error) {

		LOG_CNN_ADD_LAYERS("camada relu adicionada");
		LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                   c->camadas[c->size - 1]->saida->z);
	}
	return c->error.error;

}

int CnnAddDropOutLayer(Cnn c, double pontoAtivacao, long long int seed) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createDropOut(c->cl, sizeIn.x, sizeIn.y, sizeIn.z, pontoAtivacao, seed, entrada,
	                                        &c->error);
	c->camadas[c->size - 1]->queue = c->queue;
	if (!c->error.error) {
		LOG_CNN_ADD_LAYERS("camada dropout adicionada");
		LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                   c->camadas[c->size - 1]->saida->z);
	}
	return c->error.error;
}

int CnnAddFullConnectLayer(Cnn c, UINT tamanhoDaSaida, int funcaoDeAtivacao) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createFullConnect(c->cl, sizeIn.x, sizeIn.y, sizeIn.z, tamanhoDaSaida, entrada,
	                                            &c->parametros, funcaoDeAtivacao, 1, &c->error);
	c->camadas[c->size - 1]->queue = c->queue;
	if (!c->error.error) {
		LOG_CNN_ADD_LAYERS("camada full connect adicionada");
		LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                   c->camadas[c->size - 1]->saida->z);
	}
	return c->error.error;
}

int CnnCall(Cnn c, double *input) {
	c->error.error = TensorPutValues(c->queue, c->camadas[0]->entrada, input);
	for (int i = 0; i < c->size; ++i) {
		c->camadas[i]->ativa(c->camadas[i]);
	}

	return c->error.error;
}

int CnnLearn(Cnn c, double *target) {
	if (c->size == 0)return -1;
	Tensor lastGrad, targ;
	Tensor gradNext;
	lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
	                     c->camadas[c->size - 1]->saida->z, &c->error);
	targ = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
	                 c->camadas[c->size - 1]->saida->z, &c->error);

	clEnqueueWriteBuffer(c->queue, targ->data, CL_TRUE, 0, targ->bytes, target, 0, NULL, NULL);

	kernel_run_recursive(&c->kernelsub, c->queue, targ->x * targ->y * targ->z, max_works,
	                     &lastGrad->data, &c->camadas[c->size - 1]->saida->data, &targ->data);
	gradNext = lastGrad;
	for (int l = c->size - 1; l >= 0; l--) {
		c->camadas[l]->calc_grads(c->camadas[l], gradNext);
		if (!c->camadas[l]->flag_notlearn)
			c->camadas[l]->corrige_pesos(c->camadas[l]);
		gradNext = c->camadas[l]->gradsEntrada;
	}
	if (c->flags & CNN_FLAG_CALCULE_ERROR) {
		size_t len = lastGrad->x * lastGrad->y * lastGrad->z;
		kernel_run(&c->kernelNorm, c->queue, len, 1, &lastGrad->data, &targ->data, &len);
		clFinish(c->queue);
		clEnqueueReadBuffer(c->queue, targ->data, CL_TRUE, 0, sizeof(double), &c->normaErro, 0, NULL, NULL);
	}
	releaseTensor(&lastGrad);
	releaseTensor(&targ);
}

void cnnSave(Cnn c, FILE *dst) {
	int i;
	for (i = 0; i < c->size; ++i) {
		c->camadas[i]->salvar(c->cl, c->camadas[i], dst, &c->error);
		if (c->error.error < 0)break;
	}
	if (i != c->size) {
		if (!c->error.error) {
			c->error.error = -10;
			snprintf(c->error.msg, 255, "falha ao salvar camadas\n");
		}
	}
}

int cnnCarregar(Cnn c, FILE *src) {
	if (c->size != 0)return -1;
	Camada cm;
	Tensor entrada = NULL;
	while (1) {
		cm = carregarCamada(c->cl, src, entrada, &c->parametros, &c->error);
		if (cm == NULL) { break; }
		entrada = cm->saida;
		__addLayer(c);
		c->camadas[c->size - 1] = cm;
		cm->queue = c->queue;
		if (c->error.error < 0)break;
	}
	if (c->size > 0) {
		c->sizeIn.x = (int) c->camadas[0]->entrada->x;
		c->sizeIn.y = (int) c->camadas[0]->entrada->y;
		c->sizeIn.z = (int) c->camadas[0]->entrada->z;
	}

	return c->error.error;
}

void Cnngetout(Cnn c, double *out) {
	if (c->size < 1)return;
	clFinish(c->queue);
	clEnqueueReadBuffer(c->queue, c->camadas[c->size - 1]->saida->data, CL_TRUE, 0,
	                    c->camadas[c->size - 1]->saida->bytes, out, 0, NULL, NULL);
}

void normalizeGPU(Cnn c, double *input, double *output, int len, double maximo, double minimo) {
	if (len < 2)return;
	Tensor tinp, tout;
	double mx, mn;
	tinp = newTensor(c->cl->context, len, 1, 1, &c->error);
	tout = newTensor(c->cl->context, len, 1, 1, &c->error);
	TensorPutValues(c->queue, tinp, input);
	// achar o maximo e minimo
	kernel_run(&c->kernelfindExtreme, c->queue, len, 1, &tinp->data, &tout->data, &len);
	clFinish(c->queue);
	clEnqueueReadBuffer(c->queue, tout->data, CL_TRUE, 0, sizeof(double), &mn, 0, NULL, NULL);
	clEnqueueReadBuffer(c->queue, tout->data, CL_TRUE, sizeof(double), sizeof(double), &mx, 0, NULL, NULL);
	// nao da para normalizar
	if (mx - mn == 0.0)goto finish;
	double somador = -mn;
	double multiplicador = (maximo - minimo) / (mx - mn);
	minimo = -minimo;
	kernel_run_recursive(&c->kernelNormalize, c->queue, len, max_works, &tinp->data, &tout->data, &multiplicador,
	                     &somador, &minimo);
	clFinish(c->queue);
	TensorGetValues(c->queue, tout, output);
	finish:
	releaseTensor(&tinp);
	releaseTensor(&tout);
}

void normalizeGPUSpaceKnow(Cnn c, double *input, double *output, int len, double input_maximo, double input_minimo,
                           double maximo, double minimo) {

	Tensor tinp, tout;
	double mx, mn;
	tinp = newTensor(c->cl->context, len, 1, 1, &c->error);
	tout = newTensor(c->cl->context, len, 1, 1, &c->error);
	TensorPutValues(c->queue, tinp, input);
	// achar o maximo e minimo
	mn = input_minimo;
	mx = input_maximo;
	// nao da para normalizar
	if (mx - mn == 0.0)goto finish;
	double somador = -mn;
	double multiplicador = (maximo - minimo) / (mx - mn);
	minimo = -minimo;
	kernel_run_recursive(&c->kernelNormalize, c->queue, len, max_works, &tinp->data, &tout->data, &multiplicador,
	                     &somador, &minimo);
	clFinish(c->queue);
	TensorGetValues(c->queue, tout, output);
	finish:
	releaseTensor(&tinp);
	releaseTensor(&tout);

}

int CnnGetIndexMax(Cnn c) {
	Tensor saida = c->camadas[c->size - 1]->saida;
	Tensor entrada = c->camadas[0]->gradsEntrada;
	int len = (int) (saida->x * saida->y * saida->z);
	int error = kernel_run(&c->kernelMax, c->queue, 1, 1, &saida->data, &entrada->data, &len);
	double indice = 0;
	clEnqueueReadBuffer(c->queue, entrada->data, CL_TRUE, 0, sizeof(double), &indice, 0, NULL, NULL);
	return (int) indice;
}

#endif