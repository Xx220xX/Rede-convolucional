//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaBatchNorm.h"

const char *getCreateParamsBatchNorm(CamadaBatchNorm c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['BatchNorm',%.3g]", c->epsilon
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringBatchNorm(CamadaBatchNorm c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "BatchNorm Layer: (%u,%u,%u) -> (%u,%u,%u)\n",
	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

int batchNormRandomize(CamadaBatchNorm c, WrapperCL *cl, Exception *error);


int ativaBatchNorm(CamadaBatchNorm c) {
	/// calcular a media e o desvio padrao
	/// aplicar a normalização

	// calcula a media
	int erro;
	kernel_run_recursive(erro, c->kernelBatchNormAtiva1, c->super.queue,
	                         c->super.entrada->z,
	                         *c->super.max_works,
	                         K_ARG c->super.entrada, K_ARG c->media,
	                         K_ARG c->super.entrada->x, K_ARG c->super.entrada->y);
	if (erro)return erro;
	// calcular diferencae diferenca quadrada
	kernel_run_recursive(erro, c->kernelBatchNormAtiva2, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z, *c->super.max_works,
	                     K_ARG c->super.entrada, K_ARG c->media,
	                     K_ARG c->diferenca, K_ARG c->diferencaquad,
	                     K_ARG c->super.entrada->x, K_ARG c->super.entrada->y);

	if (erro)return erro;

	// calcula a variancia
	kernel_run_recursive(erro, c->kernelBatchNormAtiva3, c->super.queue,
	                     c->diferenca->z, *c->super.max_works,
	                     K_ARG c->diferenca, K_ARG c->diferencaquad,
	                     K_ARG c->somaDiferenca, K_ARG c->variancia, K_ARG c->epsilon,
	                     K_ARG c->diferencaquad->x, K_ARG c->diferencaquad->y);

	if (erro)return erro;
	// efetua a normalização
	kernel_run_recursive(erro, c->kernelBatchNormAtiva4, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->super.saida,
	                     K_ARG c->norma,
	                     K_ARG c->diferenca,
	                     K_ARG c->variancia,
	                     K_ARG c->Y,
	                     K_ARG c->B,
	                     K_ARG c->diferencaquad->x,
	                     K_ARG c->diferencaquad->y);
	return erro;

}

int corrige_pesosBatchNorm(CamadaBatchNorm c) {
	int erro ;
	kernel_run_recursive(erro, c->kernelBatchNormCorrige,
	                         c->super.queue,
	                         c->super.entrada->z,
	                         *c->super.max_works,
	                         K_ARG c->gradY,
	                         K_ARG c->gradB,
	                         K_ARG c->Y,
	                         K_ARG c->B,
	                         K_ARG c->super.parametros.hitLearn);
	return erro;

}

int calc_gradsBatchNorm(CamadaBatchNorm c, Tensor GradNext) {
	int erro;
	if (c->super.gradsEntrada) {
		kernel_run_recursive(erro, c->kernelBatchNormCalcGrads1, c->super.queue,
		                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
		                     *c->super.max_works,
		                     K_ARG c->super.gradsEntrada,
		                     K_ARG GradNext,
		                     K_ARG c->variancia,
		                     K_ARG c->media,
		                     K_ARG c->Y,
		                     K_ARG c->somaDiferenca,
		                     K_ARG c->super.entrada,
		                     K_ARG c->super.entrada->x,
		                     K_ARG c->super.entrada->y);
		if (erro)return erro;
	}
	kernel_run_recursive(erro, c->kernelBatchNormCalcGrads2, c->super.queue,
	                     c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG GradNext,
	                     K_ARG c->norma,
	                     K_ARG c->gradY,
	                     K_ARG c->gradB,
	                     K_ARG c->super.entrada->x,
	                     K_ARG c->super.entrada->y);
	return erro;
}

void salvarBatchNorm(WrapperCL *cl, CamadaBatchNorm c, FILE *dst, Exception *error) {
	char flag = '#';
	fwrite(& c->super.type, sizeof(char), 1, dst);
	fwrite(& c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(& flag, sizeof(char), 1, dst);
	fwrite(&  c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(& c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(& c->super.entrada->z, sizeof(UINT), 1, dst);
	double *data = callocdouble(c->Y->z);
	TensorGetValues(c->super.queue, c->Y, data);
	fwrite(data, c->Y->bytes, 1, dst);
	TensorGetValues(c->super.queue, c->B, data);
	fwrite(data, c->B->bytes, 1, dst);
	free(data);

}

Camada carregarBatchNorm(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                         Params params, Exception *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	char usehost = 0;
	fread(&usehost, sizeof(char), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	double epsilon = 1e-10;
	CamadaBatchNorm cm = (CamadaBatchNorm) createBatchNorm(cl, queue, params, inx, iny, inz, entrada,
	                                                       epsilon, 0, usehost, error);
	double *data = callocdouble(cm->Y->z);
	fread(data, cm->Y->bytes, 1, src);
	TensorPutValues(cm->super.queue, cm->Y, data);
	fread(data, cm->B->bytes, 1, src);
	TensorPutValues(cm->super.queue, cm->B, data);
	free(data);
	return (Camada) cm;
}

int batchNormRandomize(CamadaBatchNorm c, WrapperCL *cl, Exception *error) {
	UINT inz = c->super.entrada->z;
	UINT valmax = inz;
	double max_weight = 1.0 / (valmax);
	double *data = callocdouble(inz);
	for (int i = 0; i < inz; ++i) {
		data[i] = LCG_randD() * max_weight; //2.19722  (valmax) * RANDOM_BILATERAL();
	}
	error->error = TensorPutValues(c->super.queue, c->Y, data);
	if (error->error) {
		getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
		free(data);
		return error->error;
	}
	for (int i = 0; i < inz; ++i) {
		data[i] = RANDOM_BILATERAL() * max_weight; //2.19722  (valmax) * RANDOM_BILATERAL();
	}
	error->error = TensorPutValues(c->super.queue, c->B, data);
	if (error->error) {
		getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
		free(data);
		return error->error;
	}
	free(data);
	return 0;
}

void realeaseBatchNorm(CamadaBatchNorm *pc) {
	CamadaBatchNorm c = *pc;
	__releaseCamada__((Camada) c);

	releaseTensor(&c->media);
	releaseTensor(&c->somaDiferenca);
	releaseTensor(&c->variancia);
	releaseTensor(&c->gradVariancia);

	releaseTensor(&c->Y);
	releaseTensor(&c->B);
	releaseTensor(&c->gradY);
	releaseTensor(&c->gradB);
	releaseTensor(&c->diferenca);
	releaseTensor(&c->diferencaquad);
	releaseTensor(&c->norma);

	releaseKernel(&c->kernelBatchNormAtiva1);
	releaseKernel(&c->kernelBatchNormAtiva2);
	releaseKernel(&c->kernelBatchNormAtiva3);
	releaseKernel(&c->kernelBatchNormAtiva4);
	releaseKernel(&c->kernelBatchNormCalcGrads1);
	releaseKernel(&c->kernelBatchNormCalcGrads2);
	releaseKernel(&c->kernelBatchNormCorrige);
	free(c);
	*pc = NULL;
}

Camada createBatchNorm(WrapperCL *cl, cl_command_queue queue, Params params,
                       UINT inx, UINT iny,
                       UINT inz, Tensor entrada, double epsilon, int randomize,
                       char usehost, Exception *error) {
	if (error->error)return NULL;

	CamadaBatchNorm c = (CamadaBatchNorm) calloc(1, sizeof(TypecamadaBatchNorm));

	__newCamada__((Camada) c, cl, BATCHNORM, entrada, queue, params, inx, iny, inz, inx, iny, inz, usehost, error);
	c->super.toString = (cfv) tostringBatchNorm;
	c->super.getCreateParams = (cfv) getCreateParamsBatchNorm;
	c->super.release = (fv) realeaseBatchNorm;
	c->super.ativa = (fv) ativaBatchNorm;
	c->super.calc_grads = (f2v) calc_gradsBatchNorm;
	c->super.corrige_pesos = (fv) corrige_pesosBatchNorm;
	c->super.salvar = (f4v) salvarBatchNorm;

	c->media = newTensor(cl->context, queue, 1, 1, inz, usehost, error);
	c->somaDiferenca = newTensor(cl->context, queue, 1, 1, inz, usehost, error);
	c->variancia = newTensor(cl->context, queue, 1, 1, inz, usehost, error);
	c->gradVariancia = newTensor(cl->context, queue, 1, 1, inz, usehost, error);

	c->Y = newTensor(cl->context, queue, 1, 1, inz, usehost, error);
	c->B = newTensor(cl->context, queue, 1, 1, inz, usehost, error);
	c->gradY = newTensor(cl->context, queue, 1, 1, inz, usehost, error);
	c->gradB = newTensor(cl->context, queue, 1, 1, inz, usehost, error);

	c->diferenca = newTensor(cl->context, queue, inx, iny, inz, usehost, error);
	c->diferencaquad = newTensor(cl->context, queue, inx, iny, inz, usehost, error);
	c->norma = newTensor(cl->context, queue, inx, iny, inz, usehost, error);

	c->epsilon = epsilon;
	c->kernelBatchNormAtiva1 = new_Kernel(cl->program, error, BatchNormMedia, 5,
	                                      K_VOID_P, K_VOID_P,
	                                      K_INT, K_INT, K_INT);

	c->kernelBatchNormAtiva2 = new_Kernel(cl->program, error, BatchNormDiferenca, 7,
	                                      K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_INT, K_INT, K_INT);
	c->kernelBatchNormAtiva3 = new_Kernel(cl->program, error, BatchNormVariance, 8,
	                                      K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_DOUBLE, K_INT, K_INT, K_INT);
	c->kernelBatchNormAtiva4 = new_Kernel(cl->program, error, BatchNormNormaliza, 9,
	                                      K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_VOID_P, K_VOID_P,
	                                      K_INT, K_INT, K_INT);

	c->kernelBatchNormCalcGrads1 = new_Kernel(cl->program, error, BatchNormaCalcGrad1, 10,
	                                          K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                          K_VOID_P, K_VOID_P, K_VOID_P,
	                                          K_INT, K_INT, K_INT);
	c->kernelBatchNormCalcGrads2 = new_Kernel(cl->program, error, BatchNormaCalcGrad2, 7,
	                                          K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                          K_INT, K_INT, K_INT);
	c->kernelBatchNormCorrige = new_Kernel(cl->program, error, batchNormCorrigePeso, 6,
	                                       K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                       K_DOUBLE, K_INT);
	if (randomize)
		batchNormRandomize(c, cl, error);
	return (Camada) c;
}
