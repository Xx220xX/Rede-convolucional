//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaSoftMax.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../kernels/camadas/utils.h"
#include "../../kernels/camadas/softmax.h"
#endif

const char *getCreateParamsSoftMax(CamadaSoftMax c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "['SoftMax']"
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringSoftMax(CamadaSoftMax c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "SoftMax  Layer: (%u,%u,%u) -> (%u,%u,%u)\n", c->super.entrada->x, c->super.entrada->y,
					   c->super.entrada->z,
					   c->super.saida->x, c->super.saida->y, c->super.saida->z
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void realeaseSoftMax(CamadaSoftMax *pc) {
	CamadaSoftMax c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->soma);
	releaseTensor(&c->exponent);
	releaseKernel(&c->kernelSoftMaxCalcGrads);
	releaseKernel(&c->kernelSoftMaxAtiva1);
	releaseKernel(&c->kernelSoftMaxAtiva2);
	releaseKernel(&c->kernelSoftMaxAtiva3);
	free_mem(c);
	*pc = NULL;
}

int ativaSoftMax(CamadaSoftMax c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelSoftMaxAtiva1, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->super.entrada->data,
						 K_ARG c->exponent->data
						 );
	if (erro)return erro;
	kernel_run_recursive(erro, c->kernelSoftMaxAtiva2, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->exponent->data,
						 K_ARG c->soma->data,
						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y);
	if (erro)return erro;
	kernel_run_recursive(erro, c->kernelSoftMaxAtiva3, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->exponent->data,
						 K_ARG c->soma->data,
						 K_ARG c->super.saida->data,
						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y);
	return erro;
}


int calc_gradsSoftMax(CamadaSoftMax c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelSoftMaxCalcGrads, c->super.queue,
						 c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
						 *c->super.max_works,
						 K_ARG c->super.gradsEntrada->data,
						 K_ARG c->super.entrada->data,
						 K_ARG GradNext->data
	);
	return erro;

}

void salvarSoftMax(WrapperCL *cl, CamadaSoftMax c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);

}

Camada carregarSoftMax(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
					    CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createSoftMax(cl, queue, inx, iny, inz, entrada, error);
}

Camada createSoftMax(WrapperCL *cl, cl_command_queue queue, unsigned int inx, unsigned int iny,
					 unsigned int inz, Tensor entrada, CNN_ERROR *error) {
	if (error->error)return NULL;
//	fprintf(stderr, "Warning: camada softmax com falha de convergencia\n");
	CamadaSoftMax c = (CamadaSoftMax) alloc_mem(1, sizeof(TypecamadaSoftMax));

	__newCamada__((Camada) c, cl, SOFTMAX, entrada, queue, (Params) {0}, inx, iny, inz, inx, iny, inz, error);
	c->super.toString = (cfv) tostringSoftMax;
	c->super.getCreateParams = (cfv) getCreateParamsSoftMax;
	c->super.release = (fv) realeaseSoftMax;
	c->super.propagation = (fv) ativaSoftMax;
	c->super.backpropagation = (f2v) calc_gradsSoftMax;
	c->super.salvar = (f4v) salvarSoftMax;

	c->soma = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->exponent = newTensor(cl->context, queue, inx, iny, inz, 0, error);

	c->kernelSoftMaxAtiva1 = new_Kernel(cl->program, error, SoftMaxativa1, 3, K_VOID_P, K_VOID_P,K_INT);
	c->kernelSoftMaxAtiva2 = new_Kernel(cl->program, error, SoftMaxativa2, 5, K_VOID_P, K_VOID_P, K_INT,
										K_INT, K_INT);
	c->kernelSoftMaxAtiva3 = new_Kernel(cl->program, error, SoftMaxativa3, 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
										K_INT, K_INT);
	c->kernelSoftMaxCalcGrads = new_Kernel(cl->program, error, softMaxcalcgrad, 4, K_VOID_P, K_VOID_P, K_VOID_P,
										   K_INT);
	return (Camada) c;
}
