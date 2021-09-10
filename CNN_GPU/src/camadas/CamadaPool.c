//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaPool.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/pool.h"
#endif

const char *getCreateParamsPool(CamadaPool c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['Pooling',%d,%d,%d,%d]",
	                   c->passox, c->passoy, c->filtrox, c->filtroy
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringPool(CamadaPool c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Pooling  Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\tStep (%u,%u )\n"
	                   "\tFilter (%u,%u )\n",

	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->passox,c->passoy,
	                   c->filtrox,c->filtroy
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void releasePool(CamadaPool *pc) {
	CamadaPool c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->super.saida);
	releaseKernel(&c->kernelPoolCalcGrads);
	releaseKernel(&c->kernelPoolAtiva);
	free_mem(c);
	*pc = NULL;
}

int ativaPool(CamadaPool c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelPoolAtiva, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada->data,
	                     K_ARG c->super.saida->data,
	                     K_ARG c->passox, K_ARG c->passoy,
	                     K_ARG c->filtrox, K_ARG c->filtroy,

	                     K_ARG c->super.saida->x,
	                     K_ARG c->super.saida->y,
	                     K_ARG c->super.entrada->x,
	                     K_ARG c->super.entrada->y);
	return erro;
}


int calc_gradsPool(CamadaPool c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelPoolCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada->data,
	                     K_ARG c->super.gradsEntrada->data,
	                     K_ARG GradNext->data,
	                     K_ARG c->super.saida->data,
	                     K_ARG c->filtrox,
	                     K_ARG c->filtroy,
	                     K_ARG c->passox,
	                     K_ARG c->passoy,
	                     K_ARG c->super.entrada->x,
	                     K_ARG c->super.entrada->y,
	                     K_ARG c->super.saida->x,
	                     K_ARG c->super.saida->y);

	return erro;
}


void salvarPool(WrapperCL *cl, CamadaPool c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->passox, sizeof(UINT), 1, dst);
	fwrite(&c->passoy, sizeof(UINT), 1, dst);
	fwrite(&c->filtrox, sizeof(UINT), 1, dst);
	fwrite(&c->filtroy, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
}

Camada carregarPool(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
					Params params, CNN_ERROR *error) {
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT passox, passoy, filtrox, filtroy, inx, iny, inz;
	char flag_usehost;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&passox, sizeof(UINT), 1, src);
	fread(&passoy, sizeof(UINT), 1, src);
	fread(&filtrox, sizeof(UINT), 1, src);
	fread(&filtroy, sizeof(UINT), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createPool(cl, queue, passox, passoy, filtrox, filtroy, inx, iny, inz, entrada, params, flag_usehost, error);
}

Camada createPool(WrapperCL *cl, cl_command_queue queue,
				  UINT passox, UINT passoy, UINT filtrox, UINT filtroy,
				  UINT inx, UINT iny, UINT inz,
				  Tensor entrada, Params params,
				  char usehost, CNN_ERROR *error) {
	CamadaPool c = (CamadaPool) alloc_mem(1, sizeof(Typecamadapool));
	c->passox = passox;
	c->passoy = passoy;
	c->filtrox = filtrox;
	c->filtroy = filtroy;
	__newCamada__((Camada) c, cl, POOL, entrada, queue, params, inx, iny, inz,
	              (inx - filtrox) / passox + 1,
	              (iny - filtroy) / passoy + 1, inz,
	              usehost, error);
	c->super.toString = (cfv) tostringPool;
	c->super.getCreateParams = (cfv) getCreateParamsPool;
	c->super.release = (fv) releasePool;
	c->super.propagation = (fv) ativaPool;
	c->super.backpropagation = (f2v) calc_gradsPool;
	c->super.parametros = params;
	c->super.salvar = (f4v) salvarPool;

	c->kernelPoolAtiva = new_Kernel(cl->program, error, poolativa, 11,
	                                K_VOID_P, K_VOID_P,
	                                K_INT, K_INT,
	                                K_INT, K_INT,
	                                K_INT, K_INT,
	                                K_INT, K_INT, K_INT
	);
	c->kernelPoolCalcGrads = new_Kernel(cl->program, error, poolCalcGrads, 13,
	                                    K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_INT, K_INT, K_INT, K_INT,
	                                    K_INT, K_INT, K_INT, K_INT,
	                                    K_INT);
	return (Camada) c;
}
