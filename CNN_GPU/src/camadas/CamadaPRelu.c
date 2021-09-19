//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaPRelu.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/prelu.h"
#endif

const char *getCreateParamsPRelu(CamadaPRelu c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "['PRelu']"
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringPRelu(CamadaPRelu c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "PRelu  Layer: (%u,%u,%u) -> (%u,%u,%u)\n", c->super.entrada->x, c->super.entrada->y,
					   c->super.entrada->z,
					   c->super.saida->x, c->super.saida->y, c->super.saida->z
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void realeasePRelu(CamadaPRelu *pc) {
	CamadaPRelu c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->A);
	releaseTensor(&c->dA);
	releaseKernel(&c->kernelPReluCalcGrads);
	releaseKernel(&c->kernelPReluAtiva);
	free_mem(c);
	*pc = NULL;
}

int ativaPRelu(CamadaPRelu c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelPReluAtiva, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->super.entrada->data,
						 K_ARG c->super.saida->data,
						 K_ARG c->A->data
	);

	return erro;
}


int calc_gradsPRelu(CamadaPRelu c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelPReluCalcGrads, c->super.queue,
						 c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
						 *c->super.max_works,
						 K_ARG c->super.gradsEntrada->data,
						 K_ARG c->super.entrada->data,
						 K_ARG GradNext->data,
						 K_ARG c->A->data,
						 K_ARG c->dA->data,
						 K_ARG c->super.parametros.hitLearn,
						 K_ARG c->super.parametros.momento,
						 K_ARG c->super.parametros.decaimentoDePeso
	);
	return erro;

}

void salvarPRelu(WrapperCL *cl, CamadaPRelu c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->super.parametros, sizeof(Params), 1, dst);

	double *data = alloc_mem(c->A->bytes, 1);
	TensorGetValuesMem(c->super.queue, c->A, data, c->A->bytes);
	fwrite(data, c->A->bytes, 1, dst);
	free_mem(data);

}

Camada carregarPRelu(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	Params params;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#') {
		fread(&flag, sizeof(char), 1, src);
	}
	UINT inx, iny, inz;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&params, sizeof(Params), 1, src);

	CamadaPRelu c = (CamadaPRelu) createPRelu(cl, queue, inx, iny, inz, entrada, params,(RandomParam) {-1}, error);
	double *data = alloc_mem(c->A->bytes, 1);
	fread(data, c->A->bytes, 1, src);
	TensorPutValuesMem(c->super.queue, c->A, data, c->A->bytes);
	free_mem(data);
	return (Camada) c;
}

Camada createPRelu(WrapperCL *cl, cl_command_queue queue, unsigned int inx, unsigned int iny,
				   unsigned int inz, Tensor entrada,Params params,RandomParam randomParams, CNN_ERROR *error) {
	if (error->error)return NULL;

	CamadaPRelu c = (CamadaPRelu) alloc_mem(1, sizeof(TypecamadaPRelu));

	__newCamada__((Camada) c, cl, PRELU, entrada, queue, params, inx, iny, inz, inx, iny, inz, error);
	c->super.toString = (cfv) tostringPRelu;
	c->super.getCreateParams = (cfv) getCreateParamsPRelu;
	c->super.release = (fv) realeasePRelu;
	c->super.propagation = (fv) ativaPRelu;
	c->super.backpropagation = (f2v) calc_gradsPRelu;
	c->super.salvar = (f4v) salvarPRelu;
	c->A = new_Tensor(cl->context, queue, 0, c->super.entrada->x, c->super.entrada->y, c->super.entrada->z, 1,
					  error, NULL);
	c->dA = new_Tensor(cl->context, queue, 0, c->super.entrada->x, c->super.entrada->y, c->super.entrada->z, 1,
					   error, NULL);

	if (randomParams.type != -1) {
		if (randomParams.type == 0)
			TensorRandomize(queue, c->A, LCG_NORMAL, 1, 0);
		else
			TensorRandomize(queue, c->A, randomParams.type, randomParams.a, randomParams.b);
	}
	c->kernelPReluAtiva = new_Kernel(cl->program, error, preluativa, 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);

	c->kernelPReluCalcGrads = new_Kernel(cl->program, error, prelucalcgrad, 10,
										 K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
										 K_INT,
										 K_DOUBLE, K_DOUBLE, K_DOUBLE,
										 K_INT);

	return (Camada) c;
}
