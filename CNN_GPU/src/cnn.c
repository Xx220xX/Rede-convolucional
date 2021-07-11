//
// Created by Henrique on 29-May-21.
//
#include "cnn.h"

char __version__[] = "2.1.000";
char __notas__[] =
		"camada conv corrigida 2.0.007\n"
		"camada padding adicionada 2.0.009\n"
		"camada padding corrigida 2.0.011\n"
		"corrigido implementacao dropout 2.0.012\n"
		"camada polling av adicionada 2.0.013\n"
		"camada convNc adicionada 2.0.014\n"
		"Todas as camadas possui seus proprios parametros 2.0.015\n"
		"verificação interna de erros adicionada 2.0.016\n"
		"verificação de camadas 2.0.017\n"
		"Revisado todas camadas, corrigido erros internos 2.1.000\n";


const char *getVersion() {
	return __version__;
}

const char *getInfo() {
	return __notas__;
}

Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz) {
	Cnn c = (Cnn) calloc(1, sizeof(TypeCnn));
//	//int len = sprintf(c->error.context, "createCnn");
	c->parametros = p;
	c->cl = cl;
	int error = 0;
	c->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error);
	c->sizeIn = (Ponto3d) {inx, iny, inz};
	if (error) {
		c->error.error = error;
		snprintf(c->error.msg, 255, "nao foi possivel criar queue\n");
	}
	c->kernelsub = new_Kernel(cl->program, &c->error, "sub", 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	c->kerneldiv = new_Kernel(cl->program, &c->error, "div", 3, K_VOID_P, K_DOUBLE, K_INT);
	c->kerneldivInt = new_Kernel(cl->program, &c->error, "divIntDo", 4, K_VOID_P, K_VOID_P, K_DOUBLE, K_INT);
	c->kernelNorm = new_Kernel(cl->program, &c->error, "norm", 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelMax = new_Kernel(cl->program, &c->error, "maxID", 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelInt2Vector = new_Kernel(cl->program, &c->error, "int2vector", 4, K_VOID_P, K_VOID_P, K_INT, K_INT);
	c->kernelNormalize = new_Kernel(cl->program, &c->error, "normalizeVector", 6, K_VOID_P, K_VOID_P, K_DOUBLE,
	                                K_DOUBLE, K_DOUBLE,
	                                K_INT);
	c->kernelfindExtreme = new_Kernel(cl->program, &c->error, "findExtremes", 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelcreateIMG = new_Kernel(cl->program, &c->error, "createImg", 7, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT,
	                                K_INT, K_INT);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
	}
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
	releaseKernel(&c->kernelsub);
	releaseKernel(&c->kerneldiv);
	releaseKernel(&c->kerneldivInt);
	releaseKernel(&c->kernelNorm);
	releaseKernel(&c->kernelNormalize);
	releaseKernel(&c->kernelfindExtreme);
	releaseKernel(&c->kernelcreateIMG);
	releaseKernel(&c->kernelMax);
	if (c->releaseCL) {
		WrapperCL_release(c->cl);
		free(c->cl);
	}
	releaseTensor(&c->lastGrad);
	releaseTensor(&c->target);
	free(c);
	*pc = NULL;
}

Cnn createCnnWithWrapperFile(char *kernelFile, Params p, UINT inx, UINT iny, UINT inz, ULL devicetype) {
	WrapperCL *cl = (WrapperCL *) calloc(sizeof(WrapperCL), 1);
	cl->type_device = devicetype;
	WrapperCL_initbyFile(cl, kernelFile);
	Cnn c = createCnn(cl, p, inx, iny, inz);
	c->releaseCL = 1;
	return c;
}

Cnn createCnnWithWrapperProgram(const char *kernelprogram, Params p, UINT inx, UINT iny, UINT inz, ULL devicetype) {
	WrapperCL *cl = (WrapperCL *) calloc(sizeof(WrapperCL), 1);
	cl->type_device = devicetype;
	WrapperCL_init(cl, kernelprogram);
	Cnn c = createCnn(cl, p, inx, iny, inz);
	c->releaseCL = 1;
	return c;
}


Ponto3d __CnnaddLayer__(Cnn c) {
	c->size += 1;
	c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
	Ponto3d in = c->sizeIn;
	if (c->size > 1) {
		in.x = (int) c->camadas[c->size - 2]->saida->x;
		in.y = (int) c->camadas[c->size - 2]->saida->y;
		in.z = (int) c->camadas[c->size - 2]->saida->z;
	}
	releaseTensor(&c->lastGrad);
	releaseTensor(&c->target);
	return in;
}

int __CnnCheckNewLayer__(Cnn c) {
	if (c->size <= 0) {
		c->error.error = -80;
		snprintf(c->error.msg, 255, "invalid call function\n");
		return c->error.error;
	}
	if (c->error.error || c->camadas[0] == NULL) {
		c->size -= 1;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->error.error;
	}
	c->lastGrad = newTensor(c->cl->context, c->queue, c->camadas[c->size - 1]->saida->x,
	                        c->camadas[c->size - 1]->saida->y,
	                        c->camadas[c->size - 1]->saida->z, 0, &c->error);
	c->target = newTensor(c->cl->context, c->queue, c->camadas[c->size - 1]->saida->x,
	                      c->camadas[c->size - 1]->saida->y,
	                      c->camadas[c->size - 1]->saida->z, 0, &c->error);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		return c->error.error;
	}
	return 0;
}

int CnnAddConvLayer(Cnn c, char usehost, UINT passo, UINT tamanhoDoFiltro, UINT numeroDeFiltros) {
	/** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
	* em que:
	* 			S = tamanho da saida
	* 			E = tamanho da entrada
	* 			F = tamanho do filtro
	* 			Pd = preenchimento com zeros
	* 			P = passo
	**/

	if (c->error.error)return c->error.error;

//	//int len = sprintf(c->error.context, "%s", "CnnAddConvLayer");
	Ponto3d sizeIn = __CnnaddLayer__(c);
	if (!checkSizeFilter(sizeIn.x, tamanhoDoFiltro, passo) || !checkSizeFilter(sizeIn.y, tamanhoDoFiltro, passo)) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "conv: tamanho do filtro invalido\n");
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->error.error;
	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createConv(c->cl, c->queue, passo, tamanhoDoFiltro, numeroDeFiltros, sizeIn.x, sizeIn.y,
	                                     sizeIn.z,
	                                     entrada,
	                                     c->parametros, usehost, &c->error, 1);
	return __CnnCheckNewLayer__(c);
}

int CnnAddConvNcLayer(Cnn c, char usehost, UINT passox, UINT passoy, UINT largx, UINT largy, UINT filtrox, UINT filtroy,
                      UINT numeroDeFiltros) {
	/** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
	* em que:
	* 			S = tamanho da saida
	* 			E = tamanho da entrada
	* 			F = tamanho do filtro
	* 			Pd = preenchimento com zeros
	* 			P = passo
	**/
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddConvNcLayer");

	Ponto3d sizeIn = __CnnaddLayer__(c);
	if (
			((sizeIn.x - 1 - (filtrox - 1) * largx) / passox !=
			 (sizeIn.x - 1 - (filtrox - 1) * largx) / (double) passox) ||
			((sizeIn.y - 1 - (filtroy - 1) * largy) / passoy !=
			 (sizeIn.y - 1 - (filtroy - 1) * largy) / (double) passoy)
			) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "conv non-causal: tamanho do filtro invalido\n");
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));

		return c->error.error;

	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createConvNc(c->cl, c->queue, passox, passoy, largx, largy, filtrox, filtroy,
	                                       numeroDeFiltros, sizeIn.x, sizeIn.y,
	                                       sizeIn.z,
	                                       entrada,
	                                       c->parametros, usehost, &c->error, 1);
	return __CnnCheckNewLayer__(c);
}

int CnnAddPoolLayer(Cnn c, char usehost, UINT passo, UINT tamanhoDoFiltro) {
	/** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
	* em que:
	* 			S = tamanho da saida
	* 			E = tamanho da entrada
	* 			F = tamanho do filtro
	* 			Pd = preenchimento com zeros
	* 			P = passo
	**/
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddPoolLayer");

	Ponto3d sizeIn = __CnnaddLayer__(c);
	if (!checkSizeFilter(sizeIn.x, tamanhoDoFiltro, passo) || !checkSizeFilter(sizeIn.y, tamanhoDoFiltro, passo)) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "pooling(%u %u): tamanho do filtro invalido\n", passo, tamanhoDoFiltro);
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->error.error;
	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createPool(c->cl, c->queue, passo, tamanhoDoFiltro, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
	                                     c->parametros, usehost, &c->error);
	return __CnnCheckNewLayer__(c);
}

int CnnAddPoolAvLayer(Cnn c, char usehost, UINT passo, UINT tamanhoDoFiltro) {
	/** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
	* em que:
	* 			S = tamanho da saida
	* 			E = tamanho da entrada
	* 			F = tamanho do filtro
	* 			Pd = preenchimento com zeros
	* 			P = passo
	**/
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddPoolAvLayer");

	Ponto3d sizeIn = __CnnaddLayer__(c);
	if (!checkSizeFilter(sizeIn.x, tamanhoDoFiltro, passo) || !checkSizeFilter(sizeIn.y, tamanhoDoFiltro, passo)) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "average pooling(%u,%u) : tamanho do filtro invalido\n", passo, tamanhoDoFiltro);
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));

		return c->error.error;
	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createPoolAv(c->cl, c->queue, passo, tamanhoDoFiltro, sizeIn.x, sizeIn.y, sizeIn.z,
	                                       entrada,
	                                       c->parametros, usehost, &c->error);
	return __CnnCheckNewLayer__(c);
}

int CnnAddBatchNorm(Cnn c, char usehost, double epsilon) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddBatchNorm");
	Ponto3d sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createBatchNorm(c->cl, c->queue, c->parametros, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
	                                          epsilon, 1,
	                                          usehost, &c->error);
	return __CnnCheckNewLayer__(c);

}

int CnnAddReluLayer(Cnn c, char usehost) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddReluLayer");
	Ponto3d sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createRelu(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, entrada, usehost, &c->error);
	return __CnnCheckNewLayer__(c);

}

int CnnAddPaddingLayer(Cnn c, char usehost, UINT top, UINT bottom, UINT left, UINT right) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddPaddingLayer");
	Ponto3d sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] =
			createPadding(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, top, bottom, left, right, entrada,
			              usehost, &c->error);


	return __CnnCheckNewLayer__(c);
}

int CnnAddSoftMax(Cnn c, char usehost) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddSoftMax");
	Ponto3d sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createSoftMax(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
	                                        usehost, &c->error);
	return __CnnCheckNewLayer__(c);
}

int CnnAddDropOutLayer(Cnn c, char usehost, double pontoAtivacao, long long int seed) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddDropOutLayer");
	Ponto3d sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createDropOut(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, pontoAtivacao, seed, entrada,
	                                        usehost, &c->error);
	return __CnnCheckNewLayer__(c);
}

int CnnAddFullConnectLayer(Cnn c, char usehost, UINT tamanhoDaSaida, int funcaoDeAtivacao) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnAddFullConnectLayer");
	Ponto3d sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;

	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createFullConnect(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, tamanhoDaSaida, entrada,
	                                            c->parametros, funcaoDeAtivacao, 1, usehost, &c->error);
	return __CnnCheckNewLayer__(c);
}

int CnnCall(Cnn c, double *input) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "CnnCall");
	c->error.error = TensorPutValues(c->queue, c->camadas[0]->entrada, input);
	if (c->error.error)getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
	for (int i = 0; i < c->size && !c->error.error; ++i) {

		c->error.error = c->camadas[i]->ativa(c->camadas[i]);
	}

	return c->error.error;
}

int CnnLearn(Cnn c, double *target) {
	if (c->error.error)return c->error.error;
	if (c->size == 0) {
		c->error.error = -70;
		sprintf(c->error.msg, "A rede não possui nenhuma camada\n");
		return c->error.error;
	}
	Tensor lastGrad = c->lastGrad;
	Tensor targ = c->target;
	Tensor gradNext;
	c->error.error = TensorPutValues(c->queue, targ, target);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE, "CnnLearn/TensorPutValues:");
		return c->error.error;
	}

	c->error.error = kernel_run_recursive(&c->kernelsub, c->queue, targ->x * targ->y * targ->z,
	                                      c->cl->maxworks,
	                                      &lastGrad->data, &c->camadas[c->size - 1]->saida->data,
	                                      &targ->data);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE,
		                      "CnnLearn/kernel_run_recursive/kernelsub:");
		return c->error.error;
	}
	gradNext = lastGrad;
	int l;
	for (l = c->size - 1 && !c->error.error; l >= 0; l--) {

		c->error.error = c->camadas[l]->calc_grads(c->camadas[l], gradNext);
		if (!c->camadas[l]->flag_notlearn && !c->error.error) {
			c->camadas[l]->corrige_pesos(c->camadas[l]);
		}
		gradNext = c->camadas[l]->gradsEntrada;
	}
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE, "CnnLearn/%d:", l);
	}
	return c->error.error;
}

int CnnCalculeError(Cnn c) {
	if (c->error.error)return c->error.error;
	//int lenContext = sprintf(c->error.context, "%s", "CnnCalculeError");
	Tensor lastGrad = c->lastGrad;
	typetensor norma = *c->target;
	norma.bytes = sizeof(double);
	size_t len = lastGrad->x * lastGrad->y * lastGrad->z;
	c->error.error = kernel_run(&c->kernelNorm, c->queue, len, 1, &lastGrad->data, &norma.data, &len);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE, "CnnCalculeError/kernel_run:");
		return c->error.error;
	}

	c->error.error = TensorGetValues(c->queue, &norma, &c->normaErro);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE,
		                      "CnnCalculeError/TensorGetValues(%u):", norma.bytes);
		return c->error.error;
	}


	return 0;
}

void cnnSave(Cnn c, FILE *dst) {
	if (c->error.error)return;
	int i;
	for (i = 0; i < c->size; ++i) {
		c->camadas[i]->salvar(c->cl, c->camadas[i], dst, &c->error);
		if (c->error.error)break;
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
		cm = carregarCamada(c->cl, src, c->queue, entrada, c->parametros, &c->error);
		if (cm == NULL) { break; }
		entrada = cm->saida;
		__CnnaddLayer__(c);
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


void normalizeGPU(Cnn c, double *input, double *output, int len, double maximo, double minimo) {
	if (len < 2)return;
	if (c->error.error)return;
	//int lenContext = sprintf(c->error.context, "%s", "normalizeGPU");
	Tensor tinp, tout;
	double mx, mn;
	tinp = newTensor(c->cl->context, c->queue, len, 1, 1, 0, &c->error);
	tout = newTensor(c->cl->context, c->queue, len, 1, 1, 0, &c->error);
	if (c->error.error)goto finish;
	if ((c->error.error = TensorPutValues(c->queue, tinp, input)))goto finish;
	// achar o maximo e minimo
	if ((c->error.error = kernel_run(&c->kernelfindExtreme, c->queue, len, 1, &tinp->data, &tout->data, &len)))
		goto finish;
	if ((c->error.error = TensorGetValues(c->queue, tout, &mn)))goto finish;
	if ((c->error.error = TensorGetValues(c->queue, tout, &mx)))goto finish;
	// nao da para normalizar
	if (mx - mn == 0.0)goto finish;
	double somador = -mn;
	double multiplicador = (maximo - minimo) / (mx - mn);
	minimo = -minimo;
	if ((c->error.error = kernel_run_recursive(&c->kernelNormalize, c->queue, len, c->cl->maxworks, &tinp->data,
	                                           &tout->data, &multiplicador,
	                                           &somador, &minimo)))
		goto finish;
	clFinish(c->queue);
	if ((c->error.error = TensorGetValues(c->queue, tout, output)))goto finish;
	finish:
	if (c->error.error)
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
	releaseTensor(&tinp);
	releaseTensor(&tout);
}

void normalizeGPUSpaceKnow(Cnn c, double *input, double *output, int len, double input_maximo, double input_minimo,
                           double maximo, double minimo) {

	Tensor tinp, tout;
	double mx, mn;
	//int lenContext = sprintf(c->error.context, "%s", "normalizeGPUSpaceKnow");
	tinp = newTensor(c->cl->context, c->queue, len, 1, 1, 0, &c->error);
	tout = newTensor(c->cl->context, c->queue, len, 1, 1, 0, &c->error);
	if ((c->error.error = TensorPutValues(c->queue, tinp, input)))goto finish;
	// achar o maximo e minimo
	mn = input_minimo;
	mx = input_maximo;
	// nao da para normalizar
	if (mx - mn == 0.0)goto finish;
	double somador = -mn;
	double multiplicador = (maximo - minimo) / (mx - mn);
	minimo = -minimo;
	if ((c->error.error = kernel_run_recursive(&c->kernelNormalize, c->queue, len, c->cl->maxworks, &tinp->data,
	                                           &tout->data, &multiplicador,
	                                           &somador, &minimo)))
		goto finish;
	clFinish(c->queue);
	if ((c->error.error = TensorGetValues(c->queue, tout, output)))goto finish;
	finish:
	if (c->error.error)
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
	releaseTensor(&tinp);
	releaseTensor(&tout);

}

int CnnGetIndexMax(Cnn c) {
	if (c->error.error)return 0;
	//int lenContext = sprintf(c->error.context, "%s", "normalizeGPU");
	Tensor saida = c->camadas[c->size - 1]->saida;
	Tensor clmen_norma = newTensor(c->cl->context, c->queue, 1, 1, 1, 0, &c->error);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		releaseTensor(&clmen_norma);
	}
	int len = (int) (saida->x * saida->y * saida->z);
	c->error.error = kernel_run(&c->kernelMax, c->queue, 1, 1, &saida->data, &clmen_norma->data, &len);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		releaseTensor(&clmen_norma);
	}
	double indice = 0;
	c->error.error = TensorGetValues(c->queue, clmen_norma, &indice);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		releaseTensor(&clmen_norma);
	}
	releaseTensor(&clmen_norma);
	return (int) indice;
}

char *salveCnnOutAsPPMGPU(Cnn c, size_t *h_r, size_t *w_r) {
	int maxH = 0;
	int maxW = 0;
	int max_bytes = 0;
	int w, h;
	Tensor t;

	for (int cm = -1; cm < c->size; cm++) {
		if (cm == -1)
			t = c->camadas[0]->entrada;
		else
			t = c->camadas[cm]->saida;
		w = t->y;
		h = t->x;
		if (t->x > t->y) {
			w = t->x;
			h = t->y;
		}
		w = w * t->z + t->z - 1;
		maxH += h;
		if (maxW < w)maxW = w;
		if (t->bytes > max_bytes)max_bytes = t->bytes;
	}
	maxW += 2;
	maxH = maxH + c->size + 1;

	Tensor img, tout;
	double mx, mn, somador, multiplicador, minimo = 0;
	size_t len;
	tout = newTensor(c->cl->context, c->queue, max_bytes, 1, 1, 0, &c->error);
	img = newTensorChar(c->cl->context, c->queue, maxH, maxW, 1, 0, &c->error);
	int imi = 0;
	int x, y;


	for (int cm = -1; cm < c->size; cm++) {
		if (cm == -1)
			t = c->camadas[0]->entrada;
		else
			t = c->camadas[cm]->saida;
		len = t->x * t->y * t->z;
		// achar o maximo e minimo
		kernel_run(&c->kernelfindExtreme, c->queue, len, 1, &t->data, &tout->data, &len);
		clEnqueueReadBuffer(c->queue, tout->data, CL_TRUE, 0, sizeof(double), &mn, 0, NULL, NULL);
		clEnqueueReadBuffer(c->queue, tout->data, CL_TRUE, sizeof(double), sizeof(double), &mx, 0, NULL, NULL);
		// nao da para normalizar
		if (mx - mn != 0.0) {
			somador = -mn;
			multiplicador = 255 / (mx - mn);
			kernel_run_recursive(&c->kernelNormalize, c->queue, len, c->cl->maxworks, &t->data, &tout->data,
			                     &multiplicador,
			                     &somador, &minimo);
			x = t->x;
			y = t->y;
			if (t->y < t->x) {
				x = t->y;
				y = t->x;
			}
			kernel_run_recursive(&c->kernelcreateIMG, c->queue, len, c->cl->maxworks, &img->data, &tout->data, &x, &y,
			                     &imi, &maxW);
		}
		if (t->y > t->x) {
			imi += t->x;
		} else {
			imi += t->y;
		}
		imi++;
	}
	clFinish(c->queue);
	char *ans = calloc(maxH, maxW);
	*h_r = maxH;
	*w_r = maxW;
	TensorGetValues(c->queue, img, ans);
	releaseTensor(&img);
	releaseTensor(&tout);
	return ans;
}

void printCnn(Cnn c) {
	for (int i = 0; i < c->size; ++i) {
		printf("%s\n", c->camadas[i]->toString(c->camadas[i]));
	}
}

