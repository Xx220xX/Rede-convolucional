//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaSoftMax.h"
#if  defined(DISABLE_KERNELS_INSIDE_DRIVE)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/softmax.h"
#endif
const char *getCreateParamsSoftMax(CamadaSoftMax c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['SoftMax']"
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringSoftMax(CamadaSoftMax c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
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
	free(c);
	*pc = NULL;
}

int ativaSoftMax(CamadaSoftMax c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelSoftMaxAtiva1, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada->data,
	                     K_ARG c->super.saida->data);
	if (erro)return erro;
	kernel_run_recursive(erro, c->kernelSoftMaxAtiva2, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->exponent->data,
	                     K_ARG c->soma->data,
	                     K_ARG c->super.saida->data,
	                     K_ARG c->super.entrada->x,
	                     K_ARG c->super.entrada->y);
	return erro;
}

int corrige_pesosSoftMax(CamadaSoftMax c) { return 0; }

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

void salvarSoftMax(WrapperCL *cl, CamadaSoftMax c, FILE *dst, Exception *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);

}

Camada carregarSoftMax(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                       Params params, Exception *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	char flag_usehost = 0;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createSoftMax(cl, queue, inx, iny, inz, entrada, flag_usehost, error);
}

Camada createSoftMax(WrapperCL *cl, cl_command_queue queue, unsigned int inx, unsigned int iny,
                     unsigned int inz, Tensor entrada, char usehost, Exception *error) {
	if (error->error)return NULL;

	CamadaSoftMax c = (CamadaSoftMax) calloc(1, sizeof(TypecamadaSoftMax));

	__newCamada__((Camada) c, cl, SOFTMAX, entrada, queue, (Params) {0}, inx, iny, inz, inx, iny, inz, usehost, error);
	c->super.toString = (cfv) tostringSoftMax;
	c->super.getCreateParams = (cfv) getCreateParamsSoftMax;
	c->super.release = (fv) realeaseSoftMax;
	c->super.ativa = (fv) ativaSoftMax;
	c->super.calc_grads = (f2v) calc_gradsSoftMax;
	c->super.corrige_pesos = (fv) corrige_pesosSoftMax;
	c->super.salvar = (f4v) salvarSoftMax;

	c->soma = newTensor(cl->context, queue, 1, 1, inz, 0, error);
	c->exponent = newTensor(cl->context, queue, inx, iny, inz, 1, error);

	c->kernelSoftMaxAtiva1 = new_Kernel(cl->program, error, SoftMaxativa1, 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
	                                    K_INT, K_INT);
	c->kernelSoftMaxAtiva2 = new_Kernel(cl->program, error, SoftMaxativa2, 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT,
	                                    K_INT, K_INT);
	c->kernelSoftMaxCalcGrads = new_Kernel(cl->program, error, softMaxcalcgrad, 4, K_VOID_P, K_VOID_P, K_VOID_P,
	                                       K_INT);
	return (Camada) c;
}
