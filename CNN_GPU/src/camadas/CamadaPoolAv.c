//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaPoolAv.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/poolav.h"
#endif

const char *getCreateParamsPoolAv(CamadaPoolAv c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "['PoolingAv',%d,%d,%d,%d]",
					   c->passox, c->passoy, c->fx, c->fy
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
					   "\tStep (%u %u)\n",
					   "\tfilter (%u %u)\n",

					   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
					   c->super.saida->x, c->super.saida->y, c->super.saida->z,
					   c->passox, c->passoy,
					   c->fx, c->fy
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
						 K_ARG c->passox,
						 K_ARG c->passoy,
						 K_ARG c->fx,
						 K_ARG c->fy,
						 K_ARG c->super.saida->x,
						 K_ARG c->super.saida->y,
						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y);
	return erro;
}


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
						 K_ARG c->passox,
						 K_ARG c->passoy,
						 K_ARG c->fx,
						 K_ARG c->fy,
						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y,
						 K_ARG c->super.entrada->z,
						 K_ARG c->super.saida->x,
						 K_ARG c->super.saida->y);
	return erro;
}


void salvarPoolAv(WrapperCL *cl, CamadaPoolAv c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(&c->passox, sizeof(UINT), 1, dst);
	fwrite(&c->passoy, sizeof(UINT), 1, dst);
	fwrite(&c->fx, sizeof(UINT), 1, dst);
	fwrite(&c->fy, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
}

Camada carregarPoolAv(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
					  Params params, CNN_ERROR *error) {
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT px, py, fx, fy, inx, iny, inz;
	char flag_usehost = 0;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&px, sizeof(UINT), 1, src);
	fread(&py, sizeof(UINT), 1, src);
	fread(&fx, sizeof(UINT), 1, src);
	fread(&fy, sizeof(UINT), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createPoolAv(cl, queue, px, py, fx, fy, inx, iny, inz, entrada, params, flag_usehost, error);
}

Camada createPoolAv(WrapperCL *cl, QUEUE queue, UINT px, UINT py, UINT fx, UINT fy,
					UINT inx, UINT iny, UINT inz,
					Tensor entrada, Params params,
					char usehost, CNN_ERROR *error) {
	CamadaPoolAv c = (CamadaPoolAv) alloc_mem(1, sizeof(TypecamadaPoolAv));
	c->passox = px;
	c->passoy = py;
	c->fx = fx;
	c->fy = fy;
	__newCamada__((Camada) c, cl, POOLAV, entrada, queue, params, inx, iny, inz,
				  (inx - fx) / px + 1,
				  (iny - fy) / px + 1, inz,
				  usehost, error);
	c->super.toString = (cfv) tostringPoolAv;
	c->super.getCreateParams = (cfv) getCreateParamsPoolAv;
	c->super.release = (fv) releasePoolAv;
	c->super.propagation = (fv) ativaPoolAv;
	c->super.backpropagation = (f2v) calc_gradsPoolAv;
	c->super.parametros = params;
	c->super.salvar = (f4v) salvarPoolAv;

	c->kernelPoolAvAtiva = new_Kernel(cl->program, error, PoolAvativa, 11,
									  K_VOID_P, K_VOID_P,
									  K_INT, K_INT,
									  K_INT, K_INT,
									  K_INT, K_INT, K_INT, K_INT, K_INT
	);
	c->kernelPoolAvCalcGrads = new_Kernel(cl->program, error, PoolAvCalcGrads, 13,
										  K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
										  K_INT, K_INT, K_INT, K_INT,
										  K_INT, K_INT, K_INT, K_INT,
										  K_INT
	);
	return (Camada) c;
}
