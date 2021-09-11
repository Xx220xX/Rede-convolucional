//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaFullConnect.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/fullconnect.h"
#endif

const char *getCreateParamsFullConnect(CamadaFullConnect c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "['FullConnect',%d,'%s']\n",
					   c->super.saida->x * c->super.saida->y * c->super.saida->z,
					   c->fa == FTANH ? "TANH" : (c->fa == FSIGMOID ? "SIGMOID" : (c->fa == FRELU ? "RELU" : "UNKNOW"))
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringFullConnect(CamadaFullConnect c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "Full Connect Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
					   "\tactivation function %s\n",

					   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
					   c->super.saida->x, c->super.saida->y, c->super.saida->z,
					   c->fa == FTANH ? "TANH" : (c->fa == FSIGMOID ? "SIGMOID" : (c->fa == FRELU ? "RELU" : "UNKNOW"))
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void releaseFullConnect(CamadaFullConnect *pc) {
	CamadaFullConnect c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->pesos);
	releaseTensor(&c->grad);
	releaseTensor(&c->dz);
	releaseTensor(&c->z);
	releaseKernel(&c->kernelfullfixWeight);
	releaseKernel(&c->kernelfullfeed);
	releaseKernel(&c->kernelfullcalcgrad1);
	releaseKernel(&c->kernelfullcalcgrad2);
	free_mem(c);
	*pc = 0;
}


int ativaFullConnect(CamadaFullConnect c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelfullfeed, c->super.queue, c->super.saida->x,
						 *c->super.max_works,

						 K_ARG c->super.entrada->data,
						 K_ARG c->pesos->data,
						 K_ARG c->z->data,
						 K_ARG c->super.saida->data,
						 K_ARG c->fa,
						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y,
						 K_ARG c->super.entrada->z,
						 K_ARG c->pesos->x,
						 K_ARG c->pesos->y);
	return erro;

}

int corrigePesosFullConnect(CamadaFullConnect c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelfullfixWeight, c->super.queue,
						 c->pesos->x * c->pesos->y, *c->super.max_works,
						 K_ARG c->super.entrada->data,
						 K_ARG c->pesos->data,
						 K_ARG c->grad->data,
						 K_ARG c->dz->data,
						 K_ARG c->super.parametros.hitLearn,
						 K_ARG c->super.parametros.decaimentoDePeso,
						 K_ARG c->super.parametros.momento,
						 K_ARG c->pesos->y);

	return erro;
}

int calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelfullcalcgrad1, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->dz->data,
						 K_ARG GradNext->data,
						 K_ARG c->z->data,
						 K_ARG c->dfa);

	if (erro || !c->super.gradsEntrada)return erro;
	kernel_run_recursive(erro, c->kernelfullcalcgrad2, c->super.queue,
						 c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
						 *c->super.max_works,
						 K_ARG c->dz->data,
						 K_ARG c->super.gradsEntrada->data,
						 K_ARG c->pesos->data,
						 K_ARG c->pesos->x,
						 K_ARG c->pesos->y);
	if (erro)return erro;
	if (c->super.learnable)return corrigePesosFullConnect(c);
	return erro;

}

void salvarFullConnect(WrapperCL *cl, CamadaFullConnect c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->fa, sizeof(int), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->super.saida->x, sizeof(UINT), 1, dst);
	double *data = (double *) alloc_mem(c->pesos->x * c->pesos->y * c->pesos->z, sizeof(double));
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);

	TensorGetValues(queue, c->pesos, data);
	fwrite(data, 1, c->pesos->bytes, dst);
	free_mem(data);
}

Camada carregarFullConnect(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
						   Params params,
						   CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz, tamanhoSaida;
	int fa;
	fread(&fa, sizeof(int), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&tamanhoSaida, sizeof(UINT), 1, src);
	CamadaFullConnect c = (CamadaFullConnect) createFullConnect(cl, queue, inx, iny, inz, tamanhoSaida,
																entrada, params,
																fa, 0, error);
	double *data = (double *) alloc_mem(c->pesos->x * c->pesos->y * c->pesos->z, sizeof(double));
	fread(data, 1, c->pesos->bytes, src);
	TensorPutValues(queue, c->pesos, data);
	free_mem(data);
	return (Camada) c;
}

Camada createFullConnect(WrapperCL *cl, cl_command_queue queue, UINT inx, UINT iny, UINT inz, UINT tamanhoSaida,
						 Tensor entrada, Params params,
						 int funcaoDeAtivacao, int randomize,  CNN_ERROR *error) {
	if (error->error)return NULL;
	CamadaFullConnect c = (CamadaFullConnect) alloc_mem(1, sizeof(Typecamadafullconnect));
	cl_context context = cl->context;
	__newCamada__((Camada) c, cl, FULLCONNECT, entrada, queue, params, inx, iny, inz, tamanhoSaida, 1, 1, error);

	c->z = newTensor(context, queue, tamanhoSaida, 1, 1, 0, error);
	c->dz = newTensor(context, queue, tamanhoSaida, 1, 1, 0, error);
	c->pesos = newTensor(context, queue, tamanhoSaida, inx * iny * inz, 1, 0, error);
	c->grad = newTensor(context, queue, tamanhoSaida, inx * iny * inz, 1, 0, error);
	error->error = TensorFill(queue, c->grad, 0);
	if (error->error) {
		getClErrorWithContext(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE, "CreateFullConnect/TensorFill ");
	}

	if (randomize) {
		double val = 2.19722 / sqrt(c->pesos->x * c->pesos->y);
		TensorRandomize(queue, c->pesos, "uniforme", 2 * val, -val);
	}
	c->super.toString = (cfv) tostringFullConnect;
	c->super.getCreateParams = (cfv) getCreateParamsFullConnect;
	c->super.release = (fv) releaseFullConnect;
	c->super.propagation = (fv) ativaFullConnect;
	c->super.backpropagation = (f2v) calc_gradsFullConnect;
	c->fa = funcaoDeAtivacao;
	c->dfa = funcaoDeAtivacao | FLAGDIF;
	c->super.salvar = (f4v) salvarFullConnect;
	c->kernelfullfeed = new_Kernel(cl->program, error, fullfeed, 11, K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
								   K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	c->kernelfullfixWeight = new_Kernel(cl->program, error, fullfixweight, 9,
										K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
										K_DOUBLE, K_DOUBLE, K_DOUBLE,
										K_INT, K_INT);
	c->kernelfullcalcgrad1 = new_Kernel(cl->program, error, fullcalcgrads1, 5, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
										K_INT);
	c->kernelfullcalcgrad2 = new_Kernel(cl->program, error, fullcalcgrads2, 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
										K_INT,
										K_INT);
	return (Camada) c;
}

