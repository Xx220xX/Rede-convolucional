//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaRelu.h"
#if (RUN_KERNEL_USING_GPU != 1)
#include "../../kernels/camadas/utils.h"
#include "../../kernels/camadas/relu.h"
#endif
const char *getCreateParamsRelu(CamadaRelu c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['Relu']"
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringRelu(CamadaRelu c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Relu  Layer: (%u,%u,%u) -> (%u,%u,%u)\n", c->super.entrada->x, c->super.entrada->y,
	                   c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void realeaseRelu(CamadaRelu *pc) {
	CamadaRelu c = *pc;
	__releaseCamada__((Camada) c);
	releaseKernel(&c->kernelReluCalcGrads);
	releaseKernel(&c->kernelReluAtiva);
	free_mem(c);
	*pc = NULL;
}

int ativaRelu(CamadaRelu c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelReluAtiva, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada->data,
	                     K_ARG c->super.saida->data);

	return erro;
}


int calc_gradsRelu(CamadaRelu c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelReluCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG c->super.gradsEntrada->data,
	                     K_ARG c->super.entrada->data,
	                     K_ARG GradNext->data);
	return erro;

}

void salvarRelu(WrapperCL *cl, CamadaRelu c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);

}

Camada carregarRelu(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,  CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createRelu(cl, queue, inx, iny, inz, entrada, error);
}

Camada createRelu(WrapperCL *cl, cl_command_queue queue, unsigned int inx, unsigned int iny,
				  unsigned int inz, Tensor entrada, CNN_ERROR *error) {
	if (error->error)return NULL;

	CamadaRelu c = (CamadaRelu) alloc_mem(1, sizeof(TypecamadaRelu));

	__newCamada__((Camada) c, cl, RELU, entrada, queue, (Params) {0}, inx, iny, inz, inx, iny, inz,  error);
	c->super.toString = (cfv) tostringRelu;
	c->super.getCreateParams = (cfv) getCreateParamsRelu;
	c->super.release = (fv) realeaseRelu;
	c->super.propagation = (fv) ativaRelu;
	c->super.backpropagation = (f2v) calc_gradsRelu;
	c->super.salvar = (f4v) salvarRelu;

	c->kernelReluAtiva = new_Kernel(cl->program, error, reluativa, 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelReluCalcGrads = new_Kernel(cl->program, error, relucalcgrad, 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	return (Camada) c;
}
