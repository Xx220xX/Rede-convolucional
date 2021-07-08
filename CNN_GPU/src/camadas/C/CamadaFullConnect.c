//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaFullConnect.h"

const char *getCreateParamsFullConnect(CamadaFullConnect c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['FullConnect',%d,'%s']\n",
	                   c->super.saida->x*c->super.saida->y*c->super.saida->z,
	                   c->fa == FTANH ? "TANH" : (c->fa == FSIGMOID ? "SIGMOID" : (c->fa == FRELU ? "RELU" : "UNKNOW"))
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringFullConnect(CamadaFullConnect c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
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

Camada createFullConnect(WrapperCL *cl, cl_command_queue queue, UINT inx, UINT iny, UINT inz, UINT tamanhoSaida,
                         Tensor entrada, Params params,
                         int funcaoDeAtivacao, int randomize, GPU_ERROR *error) {
	if (error->error)return NULL;
	CamadaFullConnect c = (CamadaFullConnect) calloc(1, sizeof(Typecamadafullconnect));
	cl_context context = cl->context;
	__newCamada__((Camada) c, cl, FULLCONNECT, entrada, queue, params, inx, iny, inz, tamanhoSaida, 1, 1, error);

	c->z = newTensor(context,queue, tamanhoSaida, 1, 1, error);
	c->dz = newTensor(context,queue, tamanhoSaida, 1, 1, error);
	c->dz_old = newTensor(context,queue, tamanhoSaida, 1, 1, error);
	c->pesos = newTensor(context,queue, tamanhoSaida, inx * iny * inz, 1, error);

	if (randomize) {
		fullRandomize(c, cl, error);
	}
	c->super.toString = (fch) tostringFullConnect;
	c->super.getCreateParams = (fch) getCreateParamsFullConnect;
	c->super.release = (fv) releaseFullConnect;
	c->super.ativa = (fv) ativaFullConnect;
	c->super.calc_grads = (fvv) calc_gradsFullConnect;
	c->super.corrige_pesos = (fv) corrigePesosFullConnect;
	c->fa = funcaoDeAtivacao;
	c->dfa = funcaoDeAtivacao | FLAGDIF;
	c->super.salvar = (fsl) salvarFullConnect;
	c->kernelfullfeed = new_Kernel(cl->program, error, "fullfeed", 11, K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                               K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	c->kernelfullfixWeight = new_Kernel(cl->program, error, "fullfixweight", 13, K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_DOUBLE, K_DOUBLE, K_DOUBLE,
	                                    K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	c->kernelfullcalcgrad1 = new_Kernel(cl->program, error, "fullcalcgrads1", 5, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
	                                    K_INT);
	c->kernelfullcalcgrad2 = new_Kernel(cl->program, error, "fullcalcgrads2", 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
	                                    K_INT,
	                                    K_INT);
	return (Camada) c;
}

int fullRandomize(CamadaFullConnect c, WrapperCL *cl, GPU_ERROR *error) {
	unsigned int inx = c->super.entrada->x;
	unsigned int iny = c->super.entrada->y;
	unsigned int inz = c->super.entrada->z;
	unsigned int tamanhoSaida = c->super.saida->x;
	unsigned int valmax = inx * iny * inz;
	double max_weight = 1.0 / sqrt(valmax);
	//unsigned int valmax = (int) sqrt(inx * iny * inz) + 1;

	double *data = callocdouble(inx * iny * inz * tamanhoSaida);
	for (int i = 0; i < tamanhoSaida; ++i) {
		for (int j = 0; j < valmax; ++j) {
			data[TensorMap(c->pesos, i, j, 0)] =
					RANDOM_BILATERAL() * max_weight; //2.19722  (valmax) * RANDOM_BILATERAL();
		}
	}
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
	error->error = clEnqueueWriteBuffer(queue, c->pesos->data, CL_TRUE, 0, c->pesos->bytes, data, 0, NULL, NULL);
	if (error->error) {
		snprintf(error->msg, 255, "nao foi possivel copiar dados\n");
		free(data);
		clReleaseCommandQueue(queue);
		return error->error;

	}
	clFinish(queue);
	clReleaseCommandQueue(queue);
	free(data);
}


void releaseFullConnect(CamadaFullConnect *pc) {
	CamadaFullConnect c = *pc;
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
	if (c->super.__string__ != NULL) {
		free(c->super.__string__);
	}
	releaseTensor(&c->pesos);
	releaseTensor(&c->dz);
	releaseTensor(&c->dz_old);
	releaseTensor(&c->super.saida);
	releaseTensor(&c->z);
	releaseKernel(&c->kernelfullfixWeight);
	releaseKernel(&c->kernelfullfeed);
	releaseKernel(&c->kernelfullcalcgrad1);
	releaseKernel(&c->kernelfullcalcgrad2);
	free(c);
	*pc = 0;
}

int ativaFullConnect(CamadaFullConnect c) {
	int erro = kernel_run_recursive(&c->kernelfullfeed, c->super.queue, c->super.saida->x,
	                                *c->super.max_works,

	                                &c->super.entrada->data,
	                                &c->pesos->data,
	                                &c->z->data,
	                                &c->super.saida->data,
	                                &c->fa,
	                                &c->super.entrada->x,
	                                &c->super.entrada->y,
	                                &c->super.entrada->z,
	                                &c->pesos->x,
	                                &c->pesos->y);
	return erro;

}

int corrigePesosFullConnect(CamadaFullConnect c) {
	int erro = kernel_run_recursive(&c->kernelfullfixWeight, c->super.queue, c->super.saida->x, *c->super.max_works,
	                                &c->super.entrada->data,
	                                &c->pesos->data,
	                                &c->dz->data,
	                                &c->dz_old->data,
	                                &c->super.parametros.hitLearn, &c->super.parametros.decaimentoDePeso,
	                                &c->super.parametros.momento,
	                                &c->super.entrada->x,
	                                &c->super.entrada->y,
	                                &c->super.entrada->z,
	                                &c->pesos->x,
	                                &c->pesos->y);

	return erro;
}

int  calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext) {

	int erro = kernel_run_recursive(&c->kernelfullcalcgrad1, c->super.queue, c->super.saida->x, *c->super.max_works,
	                     &c->dz->data, &GradNext->data, &c->z->data, &c->dfa);
	if(erro)return erro;
	erro  = kernel_run_recursive(&c->kernelfullcalcgrad2, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->dz->data,
	                     &c->super.gradsEntrada->data,
	                     &c->pesos->data,
	                     &c->pesos->x,
	                     &c->pesos->y);
	return erro;

}

void salvarFullConnect(WrapperCL *cl, CamadaFullConnect c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->fa, sizeof(int), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->super.saida->x, sizeof(UINT), 1, dst);
	double *data = callocdouble(c->pesos->x * c->pesos->y * c->pesos->z);
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
	clEnqueueReadBuffer(queue, c->pesos->data, CL_TRUE, 0, c->pesos->bytes, data, 0, NULL, NULL);
	fwrite(data, 1, c->pesos->bytes, dst);
	clFinish(queue);
	clReleaseCommandQueue(queue);
}

Camada carregarFullConnect(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                           Params params,
                           GPU_ERROR *error) {
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
	CamadaFullConnect c = (CamadaFullConnect) createFullConnect(cl, queue, inx, iny, inz, tamanhoSaida, entrada, params,
	                                                            fa, 0,
	                                                            error);
	double *data = callocdouble(c->pesos->x * c->pesos->y * c->pesos->z);
	fread(data, 1, c->pesos->bytes, src);
	clEnqueueWriteBuffer(queue, c->pesos->data, CL_TRUE, 0, c->pesos->bytes, data, 0, NULL, NULL);
	clFinish(queue);
	return (Camada) c;
}

