//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaFullConnect.h"

const char *getCreateParamsFullConnect(CamadaFullConnect c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
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

int fullRandomize(CamadaFullConnect c, WrapperCL *cl, Exception *error) {
	unsigned int inx = c->super.entrada->x;
	unsigned int iny = c->super.entrada->y;
	unsigned int inz = c->super.entrada->z;
	unsigned int tamanhoSaida = c->super.saida->x;
	unsigned int valmax = inx * iny * inz;
	double max_weight = 1.0 / sqrt(valmax);
	double *data = callocdouble(inx * iny * inz * tamanhoSaida);
	for (int i = 0; i < tamanhoSaida; ++i) {
		for (int j = 0; j < valmax; ++j) {
			data[TensorMap(c->pesos, i, j, 0)] =
					RANDOM_BILATERAL() * max_weight; //2.19722  (valmax) * RANDOM_BILATERAL();
		}
	}
	error->error = TensorPutValues(c->super.queue, c->pesos, data);
	free(data);
	if (error->error) {
		getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
		return error->error;

	}
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
	int erro = kernel_run_recursive(&c->kernelfullfixWeight, c->super.queue,
									c->pesos->x*c->pesos->y, *c->super.max_works,
	                                &c->super.entrada->data,
	                                &c->pesos->data,
	                                &c->grad->data,
	                                &c->dz->data,
	                                &c->super.parametros.hitLearn,
	                                &c->super.parametros.decaimentoDePeso,
	                                &c->super.parametros.momento,
	                                &c->pesos->y);

	return erro;
}

int calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext) {
	int erro = kernel_run_recursive(&c->kernelfullcalcgrad1, c->super.queue,
									c->super.saida->x*c->super.saida->y*c->super.saida->z,
									*c->super.max_works,
	                                &c->dz->data, &GradNext->data, &c->z->data, &c->dfa);
	if (erro)return erro;
	if (!c->super.gradsEntrada)return 0;
	erro = kernel_run_recursive(&c->kernelfullcalcgrad2, c->super.queue,
	                            c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                            *c->super.max_works,
	                            &c->dz->data,
	                            &c->super.gradsEntrada->data,
	                            &c->pesos->data,
	                            &c->pesos->x,
	                            &c->pesos->y);
	return erro;

}

void salvarFullConnect(WrapperCL *cl, CamadaFullConnect c, FILE *dst, Exception *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(&c->fa, sizeof(int), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->super.saida->x, sizeof(UINT), 1, dst);
	double *data = callocdouble(c->pesos->x * c->pesos->y * c->pesos->z);
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);

	TensorGetValues(queue, c->pesos, data);
	fwrite(data, 1, c->pesos->bytes, dst);
	free(data);
}

Camada carregarFullConnect(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                           Params params,
                           Exception *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz, tamanhoSaida;
	int fa;
	char flag_usehost = 0;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&fa, sizeof(int), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&tamanhoSaida, sizeof(UINT), 1, src);
	CamadaFullConnect c = (CamadaFullConnect) createFullConnect(cl, queue, inx, iny, inz, tamanhoSaida,
															 entrada, params,
	                                                            fa, 0, flag_usehost, error);
	double *data = callocdouble(c->pesos->x * c->pesos->y * c->pesos->z);
	fread(data, 1, c->pesos->bytes, src);
	TensorPutValues(queue, c->pesos, data);
	free(data);
	return (Camada) c;
}

Camada createFullConnect(WrapperCL *cl, cl_command_queue queue, UINT inx, UINT iny, UINT inz, UINT tamanhoSaida,
                         Tensor entrada, Params params,
                         int funcaoDeAtivacao, int randomize, char usehost, Exception *error) {
	if (error->error)return NULL;
	CamadaFullConnect c = (CamadaFullConnect) calloc(1, sizeof(Typecamadafullconnect));
	cl_context context = cl->context;
	__newCamada__((Camada) c, cl, FULLCONNECT, entrada, queue, params, inx, iny, inz, tamanhoSaida, 1, 1, usehost,
	              error);

	c->z = newTensor(context, queue, tamanhoSaida, 1, 1, usehost, error);
	c->dz = newTensor(context, queue, tamanhoSaida, 1, 1, usehost, error);
	c->pesos = newTensor(context, queue, tamanhoSaida, inx * iny * inz, 1, usehost, error);
	c->grad = newTensor(context, queue, tamanhoSaida, inx * iny * inz, 1, usehost, error);
	error->error = TensorFill(queue,c->grad,0);
	if(error->error){
		getClErrorWithContext(error->error,error->msg,EXCEPTION_MAX_MSG_SIZE,"CreateFullConnect/TensorFill ");
	}

	if (randomize) {
		fullRandomize(c, cl, error);
	}
	c->super.toString = (cfv) tostringFullConnect;
	c->super.getCreateParams = (cfv) getCreateParamsFullConnect;
	c->super.release = (fv) releaseFullConnect;
	c->super.ativa = (fv) ativaFullConnect;
	c->super.calc_grads = (f2v) calc_gradsFullConnect;
	c->super.corrige_pesos = (fv) corrigePesosFullConnect;
	c->fa = funcaoDeAtivacao;
	c->dfa = funcaoDeAtivacao | FLAGDIF;
	c->super.salvar = (f4v) salvarFullConnect;
	c->kernelfullfeed = new_Kernel(cl->program, error, "fullfeed", 11, K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                               K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	c->kernelfullfixWeight = new_Kernel(cl->program, error, "fullfixweight", 9,
										K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_DOUBLE, K_DOUBLE, K_DOUBLE,
	                                    K_INT,  K_INT);
	c->kernelfullcalcgrad1 = new_Kernel(cl->program, error, "fullcalcgrads1", 5, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
	                                    K_INT);
	c->kernelfullcalcgrad2 = new_Kernel(cl->program, error, "fullcalcgrads2", 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
	                                    K_INT,
	                                    K_INT);
	return (Camada) c;
}

