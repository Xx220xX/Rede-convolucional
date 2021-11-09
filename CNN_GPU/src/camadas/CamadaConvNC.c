//
// Created by Henrique on 05-jul-2021.
//

#include "camadas/CamadaConvNC.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../kernels/camadas/utils.h"
#include "../../kernels/camadas/convNc.h"
#endif

const char *getCreateParamsConvNc(CamadaConvNc c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "['ConvolucaoNcausal',%d,%d,%d,%d,%d,%d,%d]",
					   c->passox, c->passoy,
					   c->largx, c->largy,
					   c->filtros->x, c->filtros->y,
					   c->numeroFiltros
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringConvNc(CamadaConvNc c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "Convolutional Non-Causal Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
					   "\tstep %u %u\n"
					   "\tlagura %u %u\n"
					   "\tfilter dim (%u %u)\n"
					   "\tnumber of filters %u\n",

					   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
					   c->super.saida->x, c->super.saida->y, c->super.saida->z,
					   c->passox, c->passoy,
					   c->largx, c->largy,
					   c->filtros->x, c->filtros->y,
					   c->numeroFiltros
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}


int ativaConvNc(CamadaConvNc c) {
	//iteraçao nos filtros
	int erro = 0;
	kernel_run_recursive(erro, c->kernelConvNcSum, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->numeroFiltros,
						 *c->super.max_works,
						 K_ARG c->filtros->data,
						 K_ARG c->super.entrada->data,
						 K_ARG c->super.saida->data,
						 K_ARG c->passox, K_ARG c->passoy, K_ARG c->largx,
						 K_ARG c->largy, K_ARG c->super.saida->x, K_ARG c->super.saida->y,
						 K_ARG c->super.entrada->x, K_ARG c->super.entrada->y,
						 K_ARG c->filtros->x, K_ARG c->filtros->y,
						 K_ARG c->super.entrada->z);
	return erro;
}

int corrige_pesosConvNc(CamadaConvNc c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelConvNcFixWeight, c->super.queue,
						 c->filtros->x * c->filtros->y * c->super.entrada->z * c->numeroFiltros,
						 *c->super.max_works,
						 K_ARG c->filtros->data,
						 K_ARG c->grad_filtros->data,
						 K_ARG c->grad_filtros_old->data,
						 K_ARG c->super.parametros.hitLearn,
						 K_ARG c->super.parametros.momento,
						 K_ARG c->super.parametros.decaimentoDePeso);
	return erro;
}

int calc_gradsConvNc(CamadaConvNc c, Tensor Gradnext) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelConvNcCalcGradsFiltro, c->super.queue,
						 c->filtros->x * c->filtros->y * c->filtros->z * c->numeroFiltros,
						 *c->super.max_works,
						 K_ARG Gradnext->data,
						 K_ARG c->super.entrada->data,
						 K_ARG c->grad_filtros->data,
						 K_ARG c->filtros->x,
						 K_ARG c->filtros->y,
						 K_ARG c->filtros->z,

						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y,

						 K_ARG c->super.saida->x,
						 K_ARG c->super.saida->y,
						 K_ARG c->passox,
						 K_ARG c->passoy,
						 K_ARG c->largx,
						 K_ARG c->largy

	);
	if (erro)return erro;
	if (!c->super.gradsEntrada)return 0;
	kernel_run_recursive(erro, c->kernelConvNcCalcGrads, c->super.queue,
						 c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
						 *c->super.max_works,

						 K_ARG c->filtros->data,
						 K_ARG c->super.entrada->data,
						 K_ARG c->super.gradsEntrada->data,
						 K_ARG Gradnext->data,

						 K_ARG c->passox,
						 K_ARG c->passoy,
						 K_ARG c->largx,
						 K_ARG c->largy,

						 K_ARG c->super.entrada->x,
						 K_ARG c->super.entrada->y,
						 K_ARG c->super.saida->x,
						 K_ARG c->super.saida->y,

						 K_ARG c->filtros->x,
						 K_ARG c->filtros->y,
						 K_ARG c->filtros->z,
						 K_ARG c->numeroFiltros);
	if (erro)return erro;
	if (c->super.learnable)return corrige_pesosConvNc(c);
	return erro;

}

void salvarConvNc(WrapperCL *cl, CamadaConvNc c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->passox, sizeof(UINT), 1, dst);
	fwrite(&c->passoy, sizeof(UINT), 1, dst);
	fwrite(&c->largx, sizeof(UINT), 1, dst);
	fwrite(&c->largy, sizeof(UINT), 1, dst);
	fwrite(&c->filtros->x, sizeof(UINT), 1, dst);
	fwrite(&c->filtros->y, sizeof(UINT), 1, dst);
	fwrite(&c->numeroFiltros, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->super.parametros, sizeof(Params), 1, dst);

	REAL *data = (REAL *) alloc_mem(c->filtros->x * c->filtros->y * c->filtros->z, sizeof(REAL));
	for (int a = 0; a < c->numeroFiltros; a++) {
		TensorGetValuesOffSet(c->super.queue, c->filtros, data, a * c->filtros->bytes);
		fwrite(data, 1, c->filtros->bytes, dst);
	}
	free_mem(data);
}

Camada carregarConvNc(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada,
					   CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	Params params;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT passox, passoy, largx, largy, fx, fy, numeroFiltros, inx, iny, inz;
	fread(&passox, sizeof(UINT), 1, src);
	fread(&passoy, sizeof(UINT), 1, src);
	fread(&fx, sizeof(UINT), 1, src);
	fread(&fy, sizeof(UINT), 1, src);
	fread(&largx, sizeof(UINT), 1, src);
	fread(&largy, sizeof(UINT), 1, src);
	fread(&numeroFiltros, sizeof(UINT), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&params, sizeof(Params), 1, src);

	CamadaConvNc c = (CamadaConvNc) createConvNc(cl, queue, passox, passoy, largx, largy, fx, fy, numeroFiltros, inx,
												 iny, inz,
												 entrada, params, (RandomParam){-1}, error);
	REAL *data = (REAL *) alloc_mem(c->filtros->x * c->filtros->y * c->super.entrada->z, sizeof(REAL));
	for (int a = 0; a < c->numeroFiltros; a++) {
		fread(data, 1, c->filtros->bytes, src);
		TensorPutValuesOffSet(queue, c->filtros, data, a * c->filtros->bytes);
	}
	free_mem(data);
	return (Camada) c;
}

void releaseConvNc(CamadaConvNc *pc) {
	CamadaConvNc c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->filtros);
	releaseTensor(&c->grad_filtros);
	releaseTensor(&c->grad_filtros_old);
	releaseKernel(&c->kernelConvNcFixWeight);
	releaseKernel(&c->kernelConvNcSum);
	releaseKernel(&c->kernelConvNcCalcGradsFiltro);
	releaseKernel(&c->kernelConvNcCalcGrads);
	free_mem(c);
	*pc = NULL;
}


Camada createConvNc(WrapperCL *cl, QUEUE queue, UINT passox,
					UINT passoy, UINT largx, UINT largy, UINT filtrox, UINT filtroy,
					UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
					Tensor entrada, Params params, RandomParam randomParams, CNN_ERROR *error) {
	if (error->error)return NULL;
	CamadaConvNc c = (CamadaConvNc) alloc_mem(1, sizeof(TypecamadaConvNc));
	__newCamada__(&c->super, cl, CONVNC, entrada, queue, params,
				  inx, iny, inz,
				  (inx - 1 - (filtrox - 1) * largx) / passox + 1,
				  (iny - 1 - (filtroy - 1) * largy) / passoy + 1,
				  numeroFiltros, error);

	c->super.toString = (cfv) tostringConvNc;
	c->super.getCreateParams = (cfv) getCreateParamsConvNc;
	c->super.release = (fv) releaseConvNc;
	c->super.propagation = (fv) ativaConvNc;
	c->super.backpropagation = (f2v) calc_gradsConvNc;
	c->super.salvar = (f4v) salvarConvNc;
	c->passox = passox;
	c->passoy = passoy;
	c->largx = largx;
	c->largy = largy;
	c->numeroFiltros = numeroFiltros;

	if (error->error)return (Camada) c;
	c->filtros = newTensor4D(cl->context, queue, filtrox, filtroy, inz, numeroFiltros, 1, error);
	c->grad_filtros = newTensor4D(cl->context, queue, filtrox, filtroy, inz, numeroFiltros, 1, error);
	c->grad_filtros_old = newTensor4D(cl->context, queue, filtrox, filtroy, inz, numeroFiltros, 1, error);


	if (randomParams.type != -1) {
		if (randomParams.type == 0)TensorRandomize(queue, c->filtros, LCG_NORMAL, 2.0 * sizeof(REAL) / (c->filtros->bytes), -1.0 * sizeof(REAL) / (c->filtros->bytes));
		else  TensorRandomize(queue, c->filtros, randomParams.type, randomParams.a,randomParams.b);
	}
	if (error->error) return (Camada) c;
	c->kernelConvNcSum = new_Kernel(cl->program, error, convncSum, 15,
									K_VOID_P, K_VOID_P, K_VOID_P,
									K_INT, K_INT, K_INT,
									K_INT, K_INT, K_INT,
									K_INT, K_INT, K_INT, K_INT,
									K_INT, K_INT);
	c->kernelConvNcFixWeight = new_Kernel(cl->program, error, convncFixWeight, 7, K_VOID_P, K_VOID_P, K_VOID_P,
										  K_REAL, K_REAL, K_REAL, K_INT);
	c->kernelConvNcCalcGradsFiltro = new_Kernel(cl->program, error, convncCalcFiltro, 15,
												K_VOID_P, K_VOID_P, K_VOID_P,
												K_INT, K_INT, K_INT,
												K_INT, K_INT, K_INT,
												K_INT, K_INT, K_INT
	);
	c->kernelConvNcCalcGrads = new_Kernel(cl->program, error, convncCalcGrads,
										  17,
										  K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
										  K_INT, K_INT, K_INT, K_INT,
										  K_INT, K_INT, K_INT, K_INT,
										  K_INT, K_INT, K_INT, K_INT,
										  K_INT);
	return (Camada) c;
}