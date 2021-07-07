//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaBatchNorm.h"

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

int batchNormRandomize(CamadaBatchNorm c, WrapperCL *cl, GPU_ERROR *error);

Camada createBatchNorm(WrapperCL *cl, cl_command_queue queue, Params params,
                       unsigned int inx, unsigned int iny,
                       unsigned int inz, Tensor entrada, double epsilon, int randomize,
                       GPU_ERROR *error) {
	if (error->error)return NULL;

	CamadaBatchNorm c = (CamadaBatchNorm) calloc(1, sizeof(TypecamadaBatchNorm));
	__newCamada__((Camada) c, cl, BATCHNORM, entrada, queue, params, inx, iny, inz, inx, iny, inz, error);
	c->super.toString = (fch) tostringBatchNorm;
	c->super.release = (fv) realeaseBatchNorm;
	c->super.ativa = (fv) ativaBatchNorm;
	c->super.calc_grads = (fvv) calc_gradsBatchNorm;
	c->super.corrige_pesos = (fv) corrige_pesosBatchNorm;
	c->super.salvar = (fsl) salvarBatchNorm;

	c->media = newTensor(cl->context, 1, 1, inz, error);
	c->somaDiferenca = newTensor(cl->context, 1, 1, inz, error);
	c->variancia = newTensor(cl->context, 1, 1, inz, error);
	c->gradVariancia = newTensor(cl->context, 1, 1, inz, error);

	c->Y = newTensor(cl->context, 1, 1, inz, error);
	c->B = newTensor(cl->context, 1, 1, inz, error);
	c->gradY = newTensor(cl->context, 1, 1, inz, error);
	c->gradB = newTensor(cl->context, 1, 1, inz, error);

	c->diferenca = newTensor(cl->context, inx, iny, inz, error);
	c->diferencaquad = newTensor(cl->context, inx, iny, inz, error);
	c->norma = newTensor(cl->context, inx, iny, inz, error);

	c->epsilon = epsilon;
	c->kernelBatchNormAtiva1 = new_Kernel(cl->program,error, "BatchNormMedia", 5,
	                                      K_VOID_P, K_VOID_P,
	                                      K_INT, K_INT, K_INT);

	c->kernelBatchNormAtiva2 = new_Kernel(cl->program, error,"BatchNormDiferenca", 7,
	                                      K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_INT, K_INT, K_INT);
	c->kernelBatchNormAtiva3 = new_Kernel(cl->program,error, "BatchNormVariance", 8,
	                                      K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_DOUBLE, K_INT, K_INT, K_INT);
	c->kernelBatchNormAtiva4 = new_Kernel(cl->program, error,"BatchNormNormaliza", 9,
	                                      K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_VOID_P, K_VOID_P,
	                                      K_INT, K_INT, K_INT);

	c->kernelBatchNormCalcGrads1 = new_Kernel(cl->program,error, "BatchNormaCalcGrad1", 10,
	                                          K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                          K_VOID_P, K_VOID_P, K_VOID_P,
	                                          K_INT, K_INT, K_INT);
	c->kernelBatchNormCalcGrads2 = new_Kernel(cl->program,error, "BatchNormaCalcGrad2", 7,
	                                          K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                          K_INT, K_INT, K_INT);
	c->kernelBatchNormCorrige = new_Kernel(cl->program,error, "batchNormCorrigePeso", 6,
	                                       K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                       K_DOUBLE, K_INT);
	if (randomize)
		batchNormRandomize(c, cl, error);
	return (Camada) c;
}

void realeaseBatchNorm(CamadaBatchNorm *pc) {
	CamadaBatchNorm c = *pc;
	releaseTensor(&c->super.gradsEntrada);
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
	releaseTensor(&c->super.saida);

	releaseTensor(&c->super.saida);

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

	if (c->super.__string__ != NULL) {
		free(c->super.__string__);
	}
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

void ativaBatchNorm(CamadaBatchNorm c) {
	/// calcular a media e o desvio padrao
	/// aplicar a normalização

	// calcula a media
	kernel_run_recursive(&c->kernelBatchNormAtiva1, c->super.queue,
	                     c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->super.entrada->data, &c->media->data, &c->super.entrada->x, &c->super.entrada->y);

	// calcular diferencae diferenca quadrada
	kernel_run_recursive(&c->kernelBatchNormAtiva2, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z, *c->super.max_works,
	                     &c->super.entrada->data, &c->media->data, &c->diferenca->data, &c->diferencaquad->data,
	                     &c->super.entrada->x, &c->super.entrada->y);



	// calcula a variancia
	kernel_run_recursive(&c->kernelBatchNormAtiva3, c->super.queue,
	                     c->diferenca->z, *c->super.max_works,
	                     &c->diferenca->data, &c->diferencaquad->data,
	                     &c->somaDiferenca->data, &c->variancia->data, &c->epsilon,
	                     &c->diferencaquad->x, &c->diferencaquad->y);


	// efetua a normalização
	kernel_run_recursive(&c->kernelBatchNormAtiva4, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     &c->super.saida->data,
	                     &c->norma->data,
	                     &c->diferenca->data,
	                     &c->variancia->data,
	                     &c->Y->data,
	                     &c->B->data,
	                     &c->diferencaquad->x,
	                     &c->diferencaquad->y);


}

void corrige_pesosBatchNorm(CamadaBatchNorm c) {

	double hit = 0.1;
	kernel_run_recursive(&c->kernelBatchNormCorrige,
	                     c->super.queue,
	                     c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->gradY->data,
	                     &c->gradB->data,
	                     &c->Y->data,
	                     &c->B->data,
	                     &hit);


}

void calc_gradsBatchNorm(CamadaBatchNorm c, Tensor GradNext) {
	kernel_run_recursive(&c->kernelBatchNormCalcGrads1, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->super.gradsEntrada->data,
	                     &GradNext->data,
	                     &c->variancia->data,
	                     &c->media->data,
	                     &c->Y->data,
	                     &c->somaDiferenca->data,
	                     &c->super.entrada->data,
	                     &c->super.entrada->x,
	                     &c->super.entrada->y);

	kernel_run_recursive(&c->kernelBatchNormCalcGrads2, c->super.queue,
	                     c->super.entrada->z,
	                     *c->super.max_works,
	                     &GradNext->data,
	                     &c->norma->data,
	                     &c->gradY->data,
	                     &c->gradB->data,
	                     &c->super.entrada->x,
	                     &c->super.entrada->y);

}

void salvarBatchNorm(WrapperCL *cl, CamadaBatchNorm c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	double *data = callocdouble(c->Y->z);
	TensorGetValues(c->super.queue, c->Y, data);
	fwrite(data, c->Y->bytes, 1, dst);
	TensorGetValues(c->super.queue, c->B, data);
	fwrite(data, c->B->bytes, 1, dst);
	free(data);

}

Camada carregarBatchNorm(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                         Params params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	double epsilon = 1e-10;
	CamadaBatchNorm cm = (CamadaBatchNorm) createBatchNorm(cl, queue, params, inx, iny, inz, entrada, epsilon, 0,
	                                                       error);
	double *data = callocdouble(cm->Y->z);
	fread(data, cm->Y->bytes, 1, src);
	TensorPutValues(cm->super.queue, cm->Y, data);
	fread(data, cm->B->bytes, 1, src);
	TensorPutValues(cm->super.queue, cm->B, data);
	free(data);
	return (Camada) cm;
}

int batchNormRandomize(CamadaBatchNorm c, WrapperCL *cl, GPU_ERROR *error) {
	unsigned int inz = c->super.entrada->z;
	unsigned int valmax = inz;
	double max_weight = 1.0 / (valmax);
	//unsigned int valmax = (int) sqrt(inx * iny * inz) + 1;

	double *data = callocdouble(inz);
	for (int i = 0; i < inz; ++i) {
		data[i] = LCG_randD() * max_weight; //2.19722  (valmax) * RANDOM_BILATERAL();
	}
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
	error->error = clEnqueueWriteBuffer(queue, c->Y->data, CL_TRUE, 0, c->Y->bytes, data, 0, NULL, NULL);
	if (error->error) {
		snprintf(error->msg, 255, "nao foi possivel copiar dados\n");
		free(data);
		clReleaseCommandQueue(queue);
		return error->error;

	}
	for (int i = 0; i < inz; ++i) {
		data[i] = RANDOM_BILATERAL() * max_weight; //2.19722  (valmax) * RANDOM_BILATERAL();
	}
	error->error = clEnqueueWriteBuffer(queue, c->B->data, CL_TRUE, 0, c->B->bytes, data, 0, NULL, NULL);
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
