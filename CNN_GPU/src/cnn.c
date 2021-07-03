//
// Created by Henrique on 29-May-21.
//
#include "cnn.h"

char __version__[] = "2.0.012";
char __notas__[] =
		"camada conv corrigida 2.0.007\n"
		"camada padding adicionada 2.0.009\n"
		"camada padding corrigida 2.0.011\n"
		"corrigido implementacao dropout 2.0.012\n"
		;

const char *getVersion() {
	return __version__;
}

const char *getInfo() {
	return __notas__;
}

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
	c->kernelsub = new_Kernel(cl->program, "sub", 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	c->kerneldiv = new_Kernel(cl->program, "div", 3, K_VOID_P, K_DOUBLE, K_INT);
	c->kerneldivInt = new_Kernel(cl->program, "divIntDo", 4, K_VOID_P, K_VOID_P, K_DOUBLE, K_INT);
	c->kernelNorm = new_Kernel(cl->program, "norm", 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelMax = new_Kernel(cl->program, "maxID", 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelInt2Vector = new_Kernel(cl->program, "int2vector", 4, K_VOID_P, K_VOID_P, K_INT, K_INT);

	c->kernelNormalize = new_Kernel(cl->program, "normalizeVector", 6, K_VOID_P, K_VOID_P, K_DOUBLE, K_DOUBLE, K_DOUBLE,
	                                K_INT);
	c->kernelfindExtreme = new_Kernel(cl->program, "findExtremes", 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelcreateIMG = new_Kernel(cl->program, "createImg", 7, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT, K_INT, K_INT);
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

Ponto3d __addLayer(Cnn c) {
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
		c->warning = INVALID_FILTER_SIZE;
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		fprintf(stderr, "tamanho do filtro invalido\n");
		return c->warning;

	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createConv(c->cl, c->queue, passo, tamanhoDoFiltro, numeroDeFiltros, sizeIn.x, sizeIn.y,
	                                     sizeIn.z,
	                                     entrada,
	                                     &c->parametros, &c->error, 1);
	if (!c->error.error) {
		c->lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                        c->camadas[c->size - 1]->saida->z, &c->error);
		c->target = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                      c->camadas[c->size - 1]->saida->z, &c->error);
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
		c->warning = INVALID_FILTER_SIZE;
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		fprintf(stderr, "tamanho do filtro invalido\n");
		return c->warning;

	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createPool(c->cl, c->queue, passo, tamanhoDoFiltro, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
	                                     &c->parametros, &c->error);
	if (!c->error.error) {
		c->lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                        c->camadas[c->size - 1]->saida->z, &c->error);
		c->target = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                      c->camadas[c->size - 1]->saida->z, &c->error);
	}
	return c->error.error;
}

int CnnAddBatchNorm(Cnn c, double epsilon) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createBatchNorm(c->cl, c->queue, &c->parametros, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
	                                          epsilon, 1,
	                                          &c->error);
	if (!c->error.error) {
		c->lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                        c->camadas[c->size - 1]->saida->z, &c->error);
		c->target = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                      c->camadas[c->size - 1]->saida->z, &c->error);
	}
	return c->error.error;

}

int CnnAddReluLayer(Cnn c) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createRelu(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, entrada, &c->error);
	if (!c->error.error) {
		c->lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                        c->camadas[c->size - 1]->saida->z, &c->error);
		c->target = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                      c->camadas[c->size - 1]->saida->z, &c->error);
	}
	return c->error.error;

}

int CnnAddPaddingLayer(Cnn c, UINT top, UINT bottom, UINT left, UINT right) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] =
			createPadding(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, top, bottom, left, right, entrada, &c->error);

	if (!c->error.error) {
		c->lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                        c->camadas[c->size - 1]->saida->z, &c->error);
		c->target = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                      c->camadas[c->size - 1]->saida->z, &c->error);
	}
	return c->error.error;

}

int CnnAddDropOutLayer(Cnn c, double pontoAtivacao, long long int seed) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createDropOut(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, pontoAtivacao, seed, entrada,
	                                        &c->error);
	if (!c->error.error) {
		c->lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                        c->camadas[c->size - 1]->saida->z, &c->error);
		c->target = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                      c->camadas[c->size - 1]->saida->z, &c->error);
	}
	return c->error.error;
}

int CnnAddFullConnectLayer(Cnn c, UINT tamanhoDaSaida, int funcaoDeAtivacao) {
	Ponto3d sizeIn = __addLayer(c);
	Tensor entrada = NULL;

	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createFullConnect(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, tamanhoDaSaida, entrada,
	                                            &c->parametros, funcaoDeAtivacao, 1, &c->error);
	if (!c->error.error) {
		c->lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                        c->camadas[c->size - 1]->saida->z, &c->error);
		c->target = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y,
		                      c->camadas[c->size - 1]->saida->z, &c->error);
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
	lastGrad = c->lastGrad;
	targ = c->target;

	clEnqueueWriteBuffer(c->queue, targ->data, CL_TRUE, 0, targ->bytes, target, 0, NULL, NULL);
	clFinish(c->queue);
	kernel_run_recursive(&c->kernelsub, c->queue, targ->x * targ->y * targ->z, c->cl->maxworks,
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
	return c->error.error;
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
		cm = carregarCamada(c->cl, src, c->queue, entrada, &c->parametros, &c->error);
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
	kernel_run_recursive(&c->kernelNormalize, c->queue, len, c->cl->maxworks, &tinp->data, &tout->data, &multiplicador,
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
	kernel_run_recursive(&c->kernelNormalize, c->queue, len, c->cl->maxworks, &tinp->data, &tout->data, &multiplicador,
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
	tout = newTensor(c->cl->context, max_bytes, 1, 1, &c->error);
	img = newTensorChar(c->cl->context, maxH, maxW, 1, &c->error);
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
		clFinish(c->queue);
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

