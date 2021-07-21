//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaRelu.h"
#include "../Camada.h"

const char *getCreateParamsRelu(CamadaRelu c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['Relu']"
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringRelu(CamadaRelu c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
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
	free(c);
	*pc = NULL;
}

int ativaRelu(CamadaRelu c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelReluAtiva, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada,
	                     K_ARG c->super.saida);

	return erro;
}

int corrige_pesosRelu(CamadaRelu c) { return 0; }

int calc_gradsRelu(CamadaRelu c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelReluCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG c->super.gradsEntrada,
	                     K_ARG c->super.entrada,
	                     K_ARG GradNext);
	return erro;

}

void salvarRelu(WrapperCL *cl, CamadaRelu c, FILE *dst, Exception *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);

}

Camada carregarRelu(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params params, Exception *error) {
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
	return createRelu(cl, queue, inx, iny, inz, entrada, flag_usehost, error);
}

Camada createRelu(WrapperCL *cl, cl_command_queue queue, unsigned int inx, unsigned int iny,
                  unsigned int inz, Tensor entrada, char usehost, Exception *error) {
	if (error->error)return NULL;

	CamadaRelu c = (CamadaRelu) calloc(1, sizeof(TypecamadaRelu));

	__newCamada__((Camada) c, cl, RELU, entrada, queue, (Params) {0}, inx, iny, inz, inx, iny, inz, usehost, error);
	c->super.toString = (cfv) tostringRelu;
	c->super.getCreateParams = (cfv) getCreateParamsRelu;
	c->super.release = (fv) realeaseRelu;
	c->super.ativa = (fv) ativaRelu;
	c->super.calc_grads = (f2v) calc_gradsRelu;
	c->super.corrige_pesos = (fv) corrige_pesosRelu;
	c->super.salvar = (f4v) salvarRelu;

	c->kernelReluAtiva = new_Kernel(cl->program, error, reluativa, 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelReluCalcGrads = new_Kernel(cl->program, error, relucalcgrad, 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	return (Camada) c;
}
