//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaPoolAv.h"
#if  defined(DISABLE_KERNELS_INSIDE_DRIVE)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/poolav.h"
#endif
const char *getCreateParamsPoolAv(CamadaPoolAv c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['PoolingAv',%d,%d]",
	                   c->passo, c->tamanhoFiltro
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringPoolAv(CamadaPoolAv c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Average Pooling  Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\tStep %u\n",

	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->passo
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void releasePoolAv(CamadaPoolAv *pc) {
	CamadaPoolAv c = *pc;
	__releaseCamada__((Camada) c);
	releaseKernel(&c->kernelPoolAvCalcGrads);
	releaseKernel(&c->kernelPoolAvAtiva);
	free_mem(c);
	*pc = NULL;
}

int ativaPoolAv(CamadaPoolAv c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelPoolAvAtiva, c->super.queue,
	                                c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                                *c->super.max_works,
	                                K_ARG c->super.entrada->data,
	                                K_ARG c->super.saida->data,
	                                K_ARG c->tamanhoFiltro,
	                                K_ARG c->passo,
	                                K_ARG c->super.saida->x, 
	                                K_ARG c->super.saida->y,
	                                K_ARG c->super.entrada->x,
	                                K_ARG c->super.entrada->y);
	return erro;
}

int corrige_pesosPoolAv(CamadaPoolAv c) { return 0; }

int calc_gradsPoolAv(CamadaPoolAv c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelPoolAvCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada->data,
	                     K_ARG c->super.gradsEntrada->data,
	                     K_ARG GradNext->data,
	                     K_ARG c->super.saida->data,
	                     K_ARG c->tamanhoFiltro,
	                     K_ARG c->passo,
	                     K_ARG c->super.entrada->x,
	                     K_ARG c->super.entrada->y,
	                     K_ARG c->super.entrada->z,
	                     K_ARG c->super.saida->x,
	                     K_ARG c->super.saida->y);
	return erro;
}


void salvarPoolAv(WrapperCL *cl, CamadaPoolAv c, FILE *dst, Exception *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(&c->passo, sizeof(UINT), 1, dst);
	fwrite(&c->tamanhoFiltro, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
}

Camada carregarPoolAv(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                      Params params, Exception *error) {
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT passo, tamanhoFiltro, inx, iny, inz;
	char flag_usehost = 0;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&passo, sizeof(UINT), 1, src);
	fread(&tamanhoFiltro, sizeof(UINT), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createPoolAv(cl, queue, passo, tamanhoFiltro, inx, iny, inz, entrada, params, flag_usehost, error);
}

Camada createPoolAv(WrapperCL *cl, cl_command_queue queue, UINT passo, UINT tamanhoFiltro, UINT inx, UINT iny, UINT inz,
                    Tensor entrada, Params params,
                    char usehost, Exception *error) {
	CamadaPoolAv c = (CamadaPoolAv) alloc_mem(1, sizeof(TypecamadaPoolAv));
	c->passo = passo;
	c->tamanhoFiltro = tamanhoFiltro;
	__newCamada__((Camada) c, cl, POOLAV, entrada, queue, params, inx, iny, inz, (inx - tamanhoFiltro) / passo + 1,
	              (iny - tamanhoFiltro) / passo + 1, inz,
	              usehost, error);
	c->super.toString = (cfv) tostringPoolAv;
	c->super.getCreateParams = (cfv) getCreateParamsPoolAv;
	c->super.release = (fv) releasePoolAv;
	c->super.ativa = (fv) ativaPoolAv;
	c->super.corrige_pesos = (fv) corrige_pesosPoolAv;
	c->super.calc_grads = (f2v) calc_gradsPoolAv;
	c->super.parametros = params;
	c->super.salvar = (f4v) salvarPoolAv;

	c->kernelPoolAvAtiva = new_Kernel(cl->program, error, PoolAvativa, 9, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT,
	                                  K_INT, K_INT,
	                                  K_INT, K_INT);
	c->kernelPoolAvCalcGrads = new_Kernel(cl->program, error, PoolAvCalcGrads, 12, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_VOID_P,
	                                      K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	return (Camada) c;
}
