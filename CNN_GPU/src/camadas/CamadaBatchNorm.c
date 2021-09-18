//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaBatchNorm.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/bathnorm.h"
#endif

const char *getCreateParamsBatchNorm(CamadaBatchNorm c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "['BatchNorm',%.3g]", c->epsilon
	);
	len += 1;
	c->super.__string__ = realloc_mem(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringBatchNorm(CamadaBatchNorm c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "BatchNorm Layer: (%u,%u,%u) -> (%u,%u,%u)\n",
					   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
					   c->super.saida->x, c->super.saida->y, c->super.saida->z
	);
	len += 1;
	c->super.__string__ = realloc_mem(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}


int ativaBatchNorm(CamadaBatchNorm c) {
	/// calcular a media e o desvio padrao
	/// aplicar a normalização

	// calcula a media
	int erro;
	kernel_run_recursive(erro, c->kernelBatchNormAtiva1, c->super.queue,
						 c->super.entrada->z,
						 *c->super.max_works,
						 K_ARG c->super.entrada->data, K_ARG c->media->data,
						 K_ARG c->super.entrada->x, K_ARG c->super.entrada->y);
	if (erro)return erro;
	// calcular diferencae diferenca quadrada
	kernel_run_recursive(erro, c->kernelBatchNormAtiva2, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z, *c->super.max_works,
						 K_ARG c->super.entrada->data, K_ARG c->media->data,
						 K_ARG c->diferenca->data, K_ARG c->diferencaquad->data,
						 K_ARG c->super.entrada->x, K_ARG c->super.entrada->y);

	if (erro)return erro;

	// calcula a variancia
	kernel_run_recursive(erro, c->kernelBatchNormAtiva3, c->super.queue,
						 c->diferenca->z, *c->super.max_works,
						 K_ARG c->diferenca->data, K_ARG c->diferencaquad->data,
						 K_ARG c->somaDiferenca->data, K_ARG c->variancia->data, K_ARG c->epsilon,
						 K_ARG c->diferencaquad->x, K_ARG c->diferencaquad->y);

	if (erro)return erro;
	// efetua a normalização
	kernel_run_recursive(erro, c->kernelBatchNormAtiva4, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->super.saida->data,
						 K_ARG c->norma->data,
						 K_ARG c->diferenca->data,
						 K_ARG c->variancia->data,
						 K_ARG c->Y->data,
						 K_ARG c->B->data,
						 K_ARG c->diferencaquad->x,
						 K_ARG c->diferencaquad->y);
	return erro;

}

int corrige_pesosBatchNorm(CamadaBatchNorm c) {
	int erro;
	kernel_run_recursive(erro, c->kernelBatchNormCorrige,
						 c->super.queue,
						 c->super.entrada->z,
						 *c->super.max_works,
						 K_ARG c->gradY->data,
						 K_ARG c->gradB->data,
						 K_ARG c->Y->data,
						 K_ARG c->B->data,
						 K_ARG c->super.parametros.hitLearn);
	return erro;

}

int calc_gradsBatchNorm(CamadaBatchNorm c, Tensor GradNext) {
	int erro;
	if (c->super.gradsEntrada) {
		kernel_run_recursive(erro, c->kernelBatchNormCalcGrads1, c->super.queue,
							 c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
							 *c->super.max_works,
							 K_ARG c->super.gradsEntrada->data,
							 K_ARG GradNext->data,
							 K_ARG c->variancia->data,
							 K_ARG c->media->data,
							 K_ARG c->Y->data,
							 K_ARG c->somaDiferenca->data,
							 K_ARG c->super.entrada,
							 K_ARG c->super.entrada->x,
							 K_ARG c->super.entrada->y);
		if (erro)return erro;
	}
	kernel_run_recursive(erro, c->kernelBatchNormCalcGrads2, c->super.queue,
						 c->super.entrada->z,
						 *c->super.max_works,
						 K_ARG GradNext->data,
						 K_ARG c->norma->data,
						 K_ARG c->gradY->data,
						 K_ARG c->gradB->data,
						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y);
	if (erro)return erro;
	if (c->super.learnable)
		erro = corrige_pesosBatchNorm(c);
	return erro;
}

void salvarBatchNorm(WrapperCL *cl, CamadaBatchNorm c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	double *data = (double *) alloc_mem(c->Y->z, sizeof(double));
	TensorGetValues(c->super.queue, c->Y, data);
	fwrite(data, c->Y->bytes, 1, dst);
	TensorGetValues(c->super.queue, c->B, data);
	fwrite(data, c->B->bytes, 1, dst);
	free_mem(data);

}

Camada carregarBatchNorm(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
						 Params params, CNN_ERROR *error) {
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
	CamadaBatchNorm cm = (CamadaBatchNorm) createBatchNorm(cl, queue, params, inx, iny, inz, entrada,
														   epsilon, (RandomParam) {-1}, (RandomParam) {-1}, error);
	double *data = (double *) alloc_mem(cm->Y->z, sizeof(double));
	fread(data, cm->Y->bytes, 1, src);
	TensorPutValues(cm->super.queue, cm->Y, data);
	fread(data, cm->B->bytes, 1, src);
	TensorPutValues(cm->super.queue, cm->B, data);
	free_mem(data);
	return (Camada) cm;
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
	free_mem(c);
	*pc = NULL;
}

Camada createBatchNorm(WrapperCL *cl, cl_command_queue queue, Params params,
					   UINT inx, UINT iny,
					   UINT inz, Tensor entrada, double epsilon, RandomParam randY, RandomParam randB,
					   CNN_ERROR *error) {
	if (error->error)return NULL;

	CamadaBatchNorm c = (CamadaBatchNorm) alloc_mem(1, sizeof(TypecamadaBatchNorm));

	__newCamada__((Camada) c, cl, BATCHNORM, entrada, queue, params, inx, iny, inz, inx, iny, inz, error);
	c->super.toString = (cfv) tostringBatchNorm;
	c->super.getCreateParams = (cfv) getCreateParamsBatchNorm;
	c->super.release = (fv) realeaseBatchNorm;
	c->super.propagation = (fv) ativaBatchNorm;
	c->super.backpropagation = (f2v) calc_gradsBatchNorm;
	c->super.salvar = (f4v) salvarBatchNorm;

	c->media = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->somaDiferenca = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->variancia = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->gradVariancia = newTensor(cl->context, queue, 1, 1, inz, 0, error);

	c->Y = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->B = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->gradY = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->gradB = newTensor(cl->context, queue, 1, 1, inz, 0, error);

	c->diferenca = newTensor(cl->context, queue, inx, iny, inz, 0, error);
	c->diferencaquad = newTensor(cl->context, queue, inx, iny, inz, 0, error);
	c->norma = newTensor(cl->context, queue, inx, iny, inz, 0, error);

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
	if (randY.type != -1) {
		if (randY.type == 0) {
			TensorRandomize(queue, c->Y, LCG_UNIFORM, 1.0 / inz, 0);
		} else
			TensorRandomize(queue, c->Y, randY.type, randY.a, randY.b);

	}
	if (randB.type != -1) {
		if (randB.type == 0) {
			TensorRandomize(queue, c->B, LCG_NORMAL, 1, 0);
		} else
			TensorRandomize(queue, c->B, randB.type, randB.a, randB.b);
	}

	return (Camada) c;
}
