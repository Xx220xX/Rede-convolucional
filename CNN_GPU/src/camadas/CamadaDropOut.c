//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaDropOut.h"
#include <time.h>
#if (RUN_KERNEL_USING_GPU != 1)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/dropout.h"
#endif
const char *getCreateParamsDropOut(CamadaDropOut c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['Dropout',%g,%zu]\n",
	                   c->p_ativacao, c->seed

	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringDropOut(CamadaDropOut c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Drop Out Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\tactivation point  %lf\n",

	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->p_ativacao

	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void releaseDropOut(CamadaDropOut *pc) {
	CamadaDropOut c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->hitmap);
	releaseKernel(&c->kerneldropcalcgrad);
	releaseKernel(&c->kerneldropativa);
	free_mem(c);
	*pc = 0;
}

int ativaDropOut(CamadaDropOut c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kerneldropativa, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada->data,
	                     K_ARG c->super.saida->data,
	                     K_ARG c->hitmap->data,
	                     K_ARG c->seed,
	                     K_ARG c->p_ativacao
	);
	c->seed += c->super.saida->x * c->super.saida->y * c->super.saida->z;
	c->seed = (c->seed * 0x5deece66dULL + 0xbULL) & ((1ULL << 31) - 1);
	return erro;
}


int calc_gradsDropOut(CamadaDropOut c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kerneldropcalcgrad, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->super.gradsEntrada->data,
	                     K_ARG c->hitmap->data,
	                     K_ARG GradNext->data);
	return erro;
}

void salvarDropOut(WrapperCL *cl, CamadaDropOut c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->p_ativacao, sizeof(double), 1, dst);

}

Camada carregarDropOut(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
					   Params params, CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	char flag_usehost = 0;
	double pativacao;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&pativacao, sizeof(double), 1, src);
	CamadaDropOut c = (CamadaDropOut) createDropOut(cl, queue, inx, iny, inz, pativacao,
	                                                time(NULL), entrada, flag_usehost, error);
	return (Camada) c;
}

Camada createDropOut(WrapperCL *cl, cl_command_queue queue, UINT inx, UINT iny, UINT inz,
					 double p_ativacao, long long seed, Tensor entrada, char usehost,
					 CNN_ERROR *error) {
	if (error->error)return NULL;
	CamadaDropOut c = (CamadaDropOut) alloc_mem(1, sizeof(Typecamadadropout));
	__newCamada__((Camada) c, cl, DROPOUT, entrada, queue, (Params) {0}, inx, iny, inz, inx, iny, inz, usehost, error);
	c->super.toString = (cfv) tostringDropOut;
	c->super.getCreateParams = (cfv) getCreateParamsDropOut;
	c->hitmap = newTensorChar(cl->context, queue, inx, iny, inz, usehost, error);
	c->p_ativacao = p_ativacao;
	c->super.release = (fv) releaseDropOut;
	c->super.propagation = (fv) ativaDropOut;
	c->super.backpropagation = (f2v) calc_gradsDropOut;
	c->seed = seed;
	c->super.salvar = (f4v) salvarDropOut;
	c->kerneldropativa = new_Kernel(cl->program, error, dropativa, 6, K_VOID_P, K_VOID_P, K_VOID_P, sizeof(cl_long),
	                                K_DOUBLE, K_INT);
	c->kerneldropcalcgrad = new_Kernel(cl->program, error, dropcalcgrad, 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	return (Camada) c;
}


