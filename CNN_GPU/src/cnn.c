//
// Created by Henrique on 29-May-21.
//
#include "cnn.h"
#include "CnnLua.h"
#include "utils/defaultkernel.h"

#if  (RUN_KERNEL_USING_GPU != 1)
#define TAG_HOST "Host mode"
#include "../kernels/camadas/utils.h"
#include "../kernels/camadas/cnnutils.h"
#else
#define TAG_HOST
#endif
#define VERSION(base, v, release) \
    int __version__ = base*10000+ v*100+ release;\
    char __strversion__[] =#base "."#v "."#release "" TAG_HOST ;

VERSION(2, 2, 19)


const char *getVersion() {
	return __strversion__;
}


void showVersion() {
	printf("##############################\n");
	printf("Gabriela IA\n");
	printf("email: gab.cnn.ia@gmail.com\n");
	printf("Versão %s\n", getVersion());
	printf("##############################\n");
}

#define CHECKDIN(input, filtro, abertura, passo) \
    (((((input-1) - (filtro - 1) * abertura) / passo +1)>0) && \
    (((((input-1) - (filtro - 1) * abertura) / passo)*passo + (filtro-1)*abertura) == (input-1)))

Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz) {
	Cnn c = (Cnn) alloc_mem(1, sizeof(Cnn_t));
	c->release_self = 1;
	c->parametros = p;
	c->cl = cl;
	int error = 0;
	c->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error);
	c->sizeIn = (Ponto) {inx, iny, inz};
	if (error) {
		c->error.error = error;
		snprintf(c->error.msg, 255, "nao foi possivel criar queue\n");
	}
	c->kernelsub = new_Kernel(cl->program, &c->error, subKernel, 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	c->kerneldiv = new_Kernel(cl->program, &c->error, divKernel, 3, K_VOID_P, K_REAL, K_INT);
	c->kerneldivInt = new_Kernel(cl->program, &c->error, divIntDo, 4, K_VOID_P, K_VOID_P, K_REAL, K_INT);
	c->kernelInt2Vector = new_Kernel(cl->program, &c->error, int2vector, 4, K_VOID_P, K_VOID_P, K_INT, K_INT);
	c->kernelNormalize = new_Kernel(cl->program, &c->error, normalizeVector, 6, K_VOID_P, K_VOID_P, K_REAL,
									K_REAL, K_REAL,K_INT);
	c->kernelcreateIMG = new_Kernel(cl->program, &c->error, createImg, 7, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT,
									K_INT, K_INT);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
	}
	c->len_input = inx * iny * inz;
	c->len_output = 0;
	return c;
}

void releaseCnn(Cnn *pc) {

	Cnn c = *pc;
	if (!c)return;
	for (int i = 0; i < c->size; ++i) {
		c->camadas[i]->release(c->camadas + i);
	}
	free_mem(c->camadas);
	clReleaseCommandQueue(c->queue);
	releaseKernel(&c->kernelsub);
	releaseKernel(&c->kerneldiv);
	releaseKernel(&c->kerneldivInt);
	releaseKernel(&c->kernelInt2Vector);
	releaseKernel(&c->kernelNormalize);
	releaseKernel(&c->kernelcreateIMG);
	if (c->releaseCL) {
		WrapperCL_release(c->cl);
		free_mem(c->cl);
	}
	releaseTensor(&c->lastGrad);
	releaseTensor(&c->target);
	if (c->L) {
		c->releaseL(c->L);
	}
	releaseDictionary(&c->luaArgs);
	if (c->release_self)
		free_mem(c);
	*pc = NULL;
}

Cnn createCnnWithWrapperFile(const char *kernelFile, Params p, UINT inx, UINT iny, UINT inz, ULL devicetype) {
	WrapperCL *cl = (WrapperCL *) alloc_mem(sizeof(WrapperCL), 1);
	cl->type_device = devicetype;
	WrapperCL_init_file(cl, kernelFile);
	Cnn c = createCnn(cl, p, inx, iny, inz);
	c->releaseCL = 1;
	return c;
}

Cnn createCnnWithWrapperProgram(const char *kernelprogram, Params p, UINT inx, UINT iny, UINT inz, ULL devicetype) {
	WrapperCL *cl = (WrapperCL *) alloc_mem(sizeof(WrapperCL), 1);
	cl->type_device = devicetype;
	if (kernelprogram == NULL)
		kernelprogram = default_kernel;
	WrapperCl_init(cl, kernelprogram);
	Cnn c = createCnn(cl, p, inx, iny, inz);
	c->releaseCL = 1;
	return c;
}

Ponto __CnnaddLayer__(Cnn c) {
	c->size += 1;
	c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
	Ponto in = c->sizeIn;
	if (c->size > 1) {
		in.x = (int) c->camadas[c->size - 2]->saida->x;
		in.y = (int) c->camadas[c->size - 2]->saida->y;
		in.z = (int) c->camadas[c->size - 2]->saida->z;
	}

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
	releaseTensor(&c->lastGrad);
	releaseTensor(&c->target);
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
	c->len_output = c->target->x * c->target->y * c->target->z;
	return 0;

}

void CnnRemoveLastLayer(Cnn c) {
	if (!c)return;
	if (c->size <= 0) return;
	c->size--;
	Tensor entrada = c->camadas[c->size]->entrada;

	c->sizeIn.x = entrada->x;
	c->sizeIn.y = entrada->y;
	c->sizeIn.z = entrada->z;
	c->camadas[c->size]->release(&c->camadas[c->size]);
	c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
	__CnnCheckNewLayer__(c);
}

int Convolucao(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT numeroDeFiltros, RandomParam randomParam) {
	if (c->error.error)return c->error.error;
	Ponto sizeIn = __CnnaddLayer__(c);
	if (!CHECKDIN(sizeIn.x, filtrox, 1, passox) || !CHECKDIN(sizeIn.x, filtroy, 1, passoy)) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "conv: tamanho do filtro invalido\n");
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->error.error;
	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createConv(c->cl, c->queue, passox, passoy, filtrox, filtroy, numeroDeFiltros, sizeIn.x, sizeIn.y, sizeIn.z, entrada, c->parametros, randomParam, &c->error);
	return __CnnCheckNewLayer__(c);
}

int ConvolucaoF(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT numeroDeFiltros, int funcAtivacao, RandomParam randomParam) {
	if (c->error.error)return c->error.error;
	Ponto sizeIn = __CnnaddLayer__(c);
	if (!CHECKDIN(sizeIn.x, filtrox, 1, passox) || !CHECKDIN(sizeIn.x, filtroy, 1, passoy)) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "conv: tamanho do filtro invalido\n");
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->error.error;
	}

	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createConvF(c->cl, c->queue, passox, passoy, filtrox, filtroy, numeroDeFiltros, sizeIn.x, sizeIn.y, sizeIn.z, funcAtivacao, entrada, c->parametros, randomParam, &c->error);
	return __CnnCheckNewLayer__(c);
}

int ConvolucaoNcausal(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT largx, UINT largy,
					  UINT numeroDeFiltros, RandomParam randomParam) {

	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "ConvolucaoNcausal");

	Ponto sizeIn = __CnnaddLayer__(c);
	if (!CHECKDIN(sizeIn.x, filtrox, largx, passox) ||
		!CHECKDIN(sizeIn.y, filtroy, largy, passoy)
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
										   c->parametros, randomParam, &c->error);
	return __CnnCheckNewLayer__(c);
}

int Pooling(Cnn c, UINT passox, UINT passoy,
			UINT filtrox, UINT filtroy) {
	if (c->error.error)return c->error.error;
	Ponto sizeIn = __CnnaddLayer__(c);
	if (!CHECKDIN(sizeIn.x, filtrox, 1, passox) ||
		!CHECKDIN(sizeIn.y, filtroy, 1, passoy)) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "pooling(%u %u %u %u): tamanho do filtro invalido\n", passox, passoy, filtrox,
				 filtroy);
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->error.error;
	}
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createPool(c->cl, c->queue, passox, passoy, filtrox, filtroy, sizeIn.x, sizeIn.y, sizeIn.z, entrada, &c->error);
	return __CnnCheckNewLayer__(c);
}

int PoolingAv(Cnn c, UINT passox, UINT pasoy,
			  UINT fx, UINT fy) {
	if (c->error.error)return c->error.error;
	Ponto sizeIn = __CnnaddLayer__(c);
	if (!CHECKDIN(sizeIn.x, fx, 1, passox) ||
		!CHECKDIN(sizeIn.y, fy, 1, pasoy)) {
		c->error.error = INVALID_FILTER_SIZE;
		snprintf(c->error.msg, 255, "average pooling(%u,%u,%u,%u) : tamanho do filtro invalido\n", passox, pasoy, fx,
				 fy);
		c->size--;
		c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
		return c->error.error;
	}
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createPoolAv(c->cl, c->queue, passox, pasoy, fx, fy, sizeIn.x, sizeIn.y, sizeIn.z, entrada, &c->error);
	return __CnnCheckNewLayer__(c);
}

int BatchNorm(Cnn c, REAL epsilon, RandomParam randomParamY, RandomParam randomParamB) {
	if (c->error.error)return c->error.error;
	Ponto sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createBatchNorm(c->cl, c->queue, c->parametros, sizeIn.x, sizeIn.y, sizeIn.z, entrada, epsilon, randomParamY, randomParamB, &c->error);
	return __CnnCheckNewLayer__(c);
}

int Relu(Cnn c, REAL lessoh, REAL greateroh) {
	if (c->error.error)return c->error.error;
	Ponto sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createRelu(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, lessoh, greateroh, entrada, &c->error);
	return __CnnCheckNewLayer__(c);
}

int PRelu(Cnn c, RandomParam randomParam) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "Relu");
	Ponto sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createPRelu(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z,
										  entrada, c->parametros, randomParam, &c->error);
	return __CnnCheckNewLayer__(c);

}

int Padding(Cnn c, UINT top, UINT bottom, UINT left, UINT right) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "Padding");
	Ponto sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] =
			createPadding(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, top, bottom, left, right, entrada,
						  &c->error);


	return __CnnCheckNewLayer__(c);
}

int SoftMax(Cnn c) {
	if (c->error.error)return c->error.error;
	Ponto sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createSoftMax(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
											&c->error);
	return __CnnCheckNewLayer__(c);
}

int Dropout(Cnn c, REAL pontoAtivacao, long long int seed) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "Dropout");
	Ponto sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;
	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createDropOut(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, pontoAtivacao, seed, entrada,
											&c->error);
	return __CnnCheckNewLayer__(c);
}

int FullConnect(Cnn c, UINT tamanhoDaSaida, int funcaoDeAtivacao, RandomParam randomParam) {
	if (c->error.error)return c->error.error;
	//int len = sprintf(c->error.context, "%s", "FullConnect");
	Ponto sizeIn = __CnnaddLayer__(c);
	Tensor entrada = NULL;

	if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
	c->camadas[c->size - 1] = createFullConnect(c->cl, c->queue, sizeIn.x, sizeIn.y, sizeIn.z, tamanhoDaSaida, entrada,
												c->parametros, funcaoDeAtivacao, randomParam, &c->error);
	return __CnnCheckNewLayer__(c);
}

int CnnCall(Cnn c, REAL *input) {
	if (c->error.error)return c->error.error;
	if (input) {
		c->error.error = TensorPutValues(c->queue, c->camadas[0]->entrada, input);
	}
	if (c->error.error)getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
	int i;
	for (i = 0; i < c->size && !c->error.error; ++i) {
		c->error.error = c->camadas[i]->propagation(c->camadas[i]);
	}
	if (c->error.error) {
		i--;
		int len = strlen(c->error.msg);
		snprintf(c->error.msg + len, EXCEPTION_MAX_MSG_SIZE - len - 1, "\nCall/camada[%d]%d/propagation", i,
				 c->camadas[i]->type);
	}
	return c->error.error;
}

int CnnLearn(Cnn c, REAL *target) {
	if (c->error.error)return c->error.error;
	if (c->size == 0) {
		c->error.error = -70;
		sprintf(c->error.msg, "A rede não possui nenhuma camada\n");
		return c->error.error;
	}
	Tensor lastGrad = c->lastGrad;
	Tensor targ = c->target;
	Tensor gradNext;
	if (target)
		c->error.error = TensorPutValues(c->queue, targ, target);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE, "CnnLearn/TensorPutValues:");
		return c->error.error;
	}

	kernel_run_recursive(c->error.error, c->kernelsub, c->queue, targ->x * targ->y * targ->z,
						 c->cl->maxworks,
						 K_ARG lastGrad->data,
						 K_ARG c->camadas[c->size - 1]->saida->data,
						 K_ARG targ->data);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE,
							  "CnnLearn/kernel_run_recursive/kernelsub:");
		return c->error.error;
	}
	gradNext = lastGrad;
	int l;
	for (l = c->size - 1; l >= 0 && !c->error.error; l--) {
		c->error.error = c->camadas[l]->backpropagation(c->camadas[l], gradNext);
		gradNext = c->camadas[l]->gradsEntrada;
	}
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE, "CnnLearn/%d/backpropagation:", l);
	}
	return c->error.error;
}

int CnnCalculeErrorWithOutput(Cnn c, REAL *target, REAL *mse) {
	if (c->error.error)return 0;
	*mse = 0;
	Tensor saida = c->camadas[c->size - 1]->saida;
	int len = (int) (saida->x * saida->y * saida->z);
	REAL *values = alloc_mem(saida->bytes, 1);
	c->error.error = TensorGetValues(c->queue, saida, values);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE,
							  "CnnCalculeErrorWithOutput/TensorGetValues ");
		free_mem(values);
		return c->error.error;
	}
	REAL sum = 0;
	REAL aux;
	for (int i = 1; i < len; i++) {
		aux = values[i] - target[i];
		sum += aux * aux;
	}
	free_mem(values);
	*mse = sqrt(sum);
	return c->error.error;
}

int CnnCalculeErrorTWithOutput(Cnn c, Tensor target, REAL *mse) {
	if (c->error.error)return 0;
	*mse = 0;
	Tensor saida = c->camadas[c->size - 1]->saida;
	int len = (int) (saida->x * saida->y * saida->z);
	REAL *values = alloc_mem(saida->bytes, 1);
	REAL *vTarget = alloc_mem(saida->bytes, 1);
	c->error.error = TensorGetValues(c->queue, saida, values);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE,
							  "CnnCalculeErrorWithOutput/TensorGetValues ");
		free_mem(vTarget);
		free_mem(values);
		return c->error.error;
	}
	c->error.error = TensorGetValues(c->queue, target, vTarget);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE,
							  "CnnCalculeErrorWithOutput/TensorGetValues ");
		free_mem(values);
		free_mem(vTarget);
		return c->error.error;
	}
	REAL sum = 0;
	REAL aux;
	for (int i = 1; i < len; i++) {
		aux = values[i] - vTarget[i];
		sum += aux * aux;
	}
	free_mem(values);
	free_mem(vTarget);
	*mse = sqrt(sum);
	return c->error.error;
}


int CnnCalculeError(Cnn c, REAL *mse) {
	if (c->error.error)return c->error.error;
	c->error.error = TensorGetNorm(c->queue, c->lastGrad, mse);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE,
							  "falha ao calcular energia do erro:");
	}
	return c->error.error;
}

int CnnGetIndexMax(Cnn c) {
	if (c->error.error)return 0;
	Tensor saida = c->camadas[c->size - 1]->saida;

	int len = (int) (saida->x * saida->y * saida->z);
	int indice = 0;
	REAL *values = alloc_mem(saida->bytes, 1);
	c->error.error = TensorGetValues(c->queue, saida, values);
	if (c->error.error) {
		getClErrorWithContext(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE, "CnnGetIndexMax/TensorGetValues ");
		free_mem(values);
		return 0;
	}
	for (int i = 1; i < len; i++) {
		if (values[i] > values[indice]) {
			indice = i;
		}
	}
	free_mem(values);
	return indice;
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
		cm = carregarCamada(c->cl, src, c->queue, entrada, &c->error);
		if (cm == NULL) { break; }
		entrada = cm->saida;
		__CnnaddLayer__(c);
		c->camadas[c->size - 1] = cm;
		__CnnCheckNewLayer__(c);
		if (c->error.error < 0)break;
		cm = NULL;
	}
	if (c->size > 0) {
		c->sizeIn.x = (int) c->camadas[c->size - 1]->entrada->x;
		c->sizeIn.y = (int) c->camadas[c->size - 1]->entrada->y;
		c->sizeIn.z = (int) c->camadas[c->size - 1]->entrada->z;
		c->len_input = c->camadas[0]->entrada->x * c->camadas[0]->entrada->y * c->camadas[0]->entrada->z;
		c->len_output = c->camadas[c->size - 1]->saida->x * c->camadas[c->size - 1]->saida->y * c->camadas[c->size - 1]->saida->z;
	}
	return c->error.error;
}

void normalizeGPU(Cnn c, REAL *input, REAL *output, int len, REAL maximo, REAL minimo) {
	if (len < 2)return;
	if (c->error.error)return;
	// achar o maximo e minimo
	REAL mx = input[0];
	REAL mn = input[0];
	for (int i = len - 1; i > 0; i--) {
		if (input[i] > mx) {
			mx = input[i];
		}
		if (input[i] < mn) {
			mn = input[i];
		}
	}
	normalizeGPUSpaceKnow(c, input, output, len, mx, mn, maximo, minimo);
}

void normalizeGPUSpaceKnow(Cnn c, REAL *input, REAL *output, int len, REAL input_maximo, REAL input_minimo,
						   REAL maximo, REAL minimo) {

	Tensor tinp, tout;
	REAL mx, mn;
	//int lenContext = sprintf(c->error.context, "%s", "normalizeGPUSpaceKnow");
	tinp = newTensor(c->cl->context, c->queue, len, 1, 1, 0, &c->error);
	tout = tinp;//newTensor(c->cl->context, c->queue, len, 1, 1, 0, &c->error);
	if ((c->error.error = TensorPutValues(c->queue, tinp, input)))goto finish;
	// achar o maximo e minimo
	mn = input_minimo;
	mx = input_maximo;
	// nao da para normalizar
	if (mx - mn == 0.0)goto finish;
	REAL somador = -mn;
	REAL multiplicador = (maximo - minimo) / (mx - mn);
	minimo = -minimo;
	kernel_run_recursive(c->error.error, c->kernelNormalize, c->queue, len, c->cl->maxworks,
						 K_ARG tinp->data,
						 K_ARG tout->data,
						 K_ARG multiplicador,
						 K_ARG somador,
						 K_ARG minimo);

	if (c->error.error)
		goto finish;
	if ((c->error.error = TensorGetValues(c->queue, tout, output)))goto finish;

	finish:
	if (c->error.error)
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
	if (tinp != tout)
		releaseTensor(&tout);

	releaseTensor(&tinp);

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
	REAL mx, mn, somador, multiplicador, minimo = 0;
	size_t len;
	tout = newTensor(c->cl->context, c->queue, max_bytes, 1, 1, 0, &c->error);
	img = newTensorChar(c->cl->context, c->queue, maxH, maxW, 1, 0, &c->error);
	int imi = 0;
	int x, y;
	REAL *values = alloc_mem(max_bytes, 1);
	int erro;
	for (int cm = -1; cm < c->size; cm++) {
		if (cm == -1)
			t = c->camadas[0]->entrada;
		else
			t = c->camadas[cm]->saida;
		len = t->x * t->y * t->z;
		// achar o maximo e minimo
		values = (REAL *) alloc_mem(len, sizeof(REAL));
		if (TensorGetValues(c->queue, t, values))continue;
		mx = values[0];
		mn = values[0];
		for (int i = len - 1; i > 0; i--) {
			if (values[i] > mx) {
				mx = values[i];
			}
			if (values[i] < mn) {
				mn = values[i];
			}
		}
		// nao da para normalizar
		if (mx - mn != 0.0) {
			somador = -mn;
			multiplicador = 255 / (mx - mn);
			kernel_run_recursive(erro, c->kernelNormalize, c->queue, len, c->cl->maxworks,
								 K_ARG t->data,
								 K_ARG tout->data,
								 K_ARG multiplicador,
								 K_ARG somador,
								 K_ARG minimo);
			x = t->x;
			y = t->y;
			if (t->y < t->x) {
				x = t->y;
				y = t->x;
			}
			kernel_run_recursive(erro, c->kernelcreateIMG, c->queue, len, c->cl->maxworks,
								 K_ARG img->data,
								 K_ARG tout->data,
								 K_ARG x,
								 K_ARG y,
								 K_ARG imi,
								 K_ARG maxW);
		}
		if (t->y > t->x) {
			imi += t->x;
		} else {
			imi += t->y;
		}
		imi++;
	}
	clFinish(c->queue);
	free_mem(values);
	char *ans = alloc_mem(maxH, maxW);
	*h_r = maxH;
	*w_r = maxW;
	TensorGetValues(c->queue, img, ans);
	releaseTensor(&img);
	releaseTensor(&tout);
	return ans;
}

void printCnn(Cnn c) {
	printf("CNN:\n");
	for (int i = 0; i < c->size; ++i) {
		printf("%s\n", c->camadas[i]->toString(c->camadas[i]));
	}
}

void CnnInitLuaVm(Cnn c) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	loadCnnLuaLibrary(L);
	lua_pushlightuserdata(L, c);
	lua_setglobal(L, LCNN);
	c->L = L;
	c->releaseL = (fv) lua_close;

}

int CnnCallT(Cnn c, Tensor input) {
	if (!c)return NULL_PARAM;
	if (!input)return NULL_PARAM;
	if (!c->size)return NULL_PARAM;
	int erro = 0;
	Tensor aux;
	switch (input->flag & TENSOR_MASK_MEM) {
		case TENSOR_SVM:
		case TENSOR_RAM:
			return CnnCall(c, input->hostd);
		case TENSOR_GPU:
			aux = c->camadas[0]->entrada;
			c->camadas[0]->entrada = input;
			erro = CnnCall(c, NULL);
			c->camadas[0]->entrada = aux;
			return erro;
		default:
			return TENSOR_INVALID_FLAG_MEM;
	}
}

int CnnLearnT(Cnn c, Tensor target) {
	if (!c)return NULL_PARAM;
	if (!target)return NULL_PARAM;
	if (!c->size)return NULL_PARAM;
	int erro = 0;
	Tensor aux;
	switch (target->flag & TENSOR_MASK_MEM) {
		case TENSOR_SVM:
		case TENSOR_RAM:
			return CnnLearn(c, target->hostd);
		case TENSOR_GPU:
			aux = c->target;
			c->target = target;
			erro = CnnLearn(c, NULL);
			c->target = aux;
			return erro;
		default:
			c->error.error = TENSOR_INVALID_FLAG_MEM;
			return TENSOR_INVALID_FLAG_MEM;
	}
}
