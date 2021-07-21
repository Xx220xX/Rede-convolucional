//
// Created by Henrique on 5/8/2021.
//

#include "../CamadaConv.h"

const char *getCreateParamsConv(CamadaConv c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                  "['Convolucao',%d,%d,%d]",
	                   c->passo,
	                   c->filtros->x,
	                   c->filtros->w
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringConv(CamadaConv c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Convolutional Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\tstep %u\n"
	                   "\tfilter dim (%u %u)\n"
	                   "\tnumber of filters %u\n",

	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->passo,
	                   c->filtros->x, c->filtros->y,
	                   c->filtros->w
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}


int convRandomize(CamadaConv c, WrapperCL *cl, Exception *error) {
	int inz = c->super.entrada->z;
	double maxVal = 1.0 / (double) (c->filtros->x * c->filtros->y * c->filtros->z);

	double *data = (double *) calloc(c->filtros->x * c->filtros->y * c->filtros->z, sizeof(double));
	for (int a = 0; a < c->filtros->w; a++) {
		FOR3D(i, j, z, c->filtros->x, c->filtros->y, inz) {
					data[TensorMap(c->filtros, i, j, z)] = RANDOM_BILATERAL() * maxVal;
				}
		error->error = TensorPutValuesOffSet(c->super.queue, c->filtros, data, a * c->filtros->bytes);
		if (error->error) {
			getClErrorWithContext(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE,
			                      "convRandomize/TensorPutValuesOffSet" );
			free(data);
			return error->error;
		}
	}
	free(data);
	return 0;
}


int ativaConv(CamadaConv c) {
	//iteraçao nos filtros
	int erro = 0;
	kernel_run_recursive(erro, c->kernelConvSum, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     K_ARG c->filtros, K_ARG c->super.entrada, K_ARG c->super.saida,
	                     K_ARG c->passo, K_ARG c->passo,
	                     K_ARG c->super.saida->x, K_ARG c->super.saida->y,
	                     K_ARG c->super.entrada->x, K_ARG c->super.entrada->y,
	                     K_ARG c->filtros->x, K_ARG c->filtros->y, K_ARG c->filtros->z);
	return erro;
}

int corrige_pesosConv(CamadaConv c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelConvFixWeight, c->super.queue,
	                     c->filtros->x * c->filtros->y * c->filtros->z * c->filtros->w,
	                     *c->super.max_works,
	                     K_ARG c->filtros,
	                     K_ARG c->gradnext,
	                     K_ARG c->super.entrada,
	                     K_ARG c->grad_filtros,
	                     K_ARG c->filtros->x, K_ARG c->filtros->y, K_ARG c->filtros->z,
	                     K_ARG c->super.entrada->x, K_ARG c->super.entrada->y,
	                     K_ARG c->super.saida->x, K_ARG c->super.saida->y,
	                     K_ARG c->passo, K_ARG c->passo,
	                     K_ARG c->super.parametros.hitLearn,
	                     K_ARG c->super.parametros.momento,
	                     K_ARG c->super.parametros.decaimentoDePeso);
	return erro;
}

int calc_gradsConv(CamadaConv c, Tensor Gradnext) {
	c->gradnext = Gradnext;
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelConvCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG c->filtros,
	                     K_ARG c->super.gradsEntrada,
	                     K_ARG Gradnext,
	                     K_ARG c->filtros->x, K_ARG c->filtros->y, K_ARG c->filtros->z,
	                     K_ARG c->passo, K_ARG c->passo,
	                     K_ARG c->super.entrada->x, K_ARG c->super.entrada->y,
	                     K_ARG c->super.saida->x, K_ARG c->super.saida->y, K_ARG c->super.saida->z);
	return erro;


}

void salvarConv(WrapperCL *cl, CamadaConv c, FILE *dst, Exception *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->passo, sizeof(UINT), 1, dst);
	fwrite(&c->filtros->x, sizeof(UINT), 1, dst);
	fwrite(&c->filtros->w, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	double *data = callocdouble(c->filtros->x * c->filtros->y * c->super.entrada->z);
	for (int a = 0; a < c->filtros->w; a++) {
		error->error = TensorGetValuesOffSet(c->super.queue, c->filtros, data, a * c->filtros->bytes);
		if (error->error) {
			getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
			break;
		}
		fwrite(data, 1, c->filtros->bytes, dst);
	}
	free(data);
}

Camada carregarConv(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada,
                    Params params, Exception *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT passo, tamanhoFiltro, numeroFiltros, inx, iny, inz;
	char flag_usehost = 0;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&passo, sizeof(UINT), 1, src);
	fread(&tamanhoFiltro, sizeof(UINT), 1, src);
	fread(&numeroFiltros, sizeof(UINT), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	CamadaConv c = (CamadaConv) createConv(cl, queue, passo, tamanhoFiltro, numeroFiltros, inx, iny, inz, entrada,
	                                       params, flag_usehost, error, 0);
	double *data = callocdouble(c->filtros->x * c->filtros->y * c->super.entrada->z);
	for (int a = 0; a < c->filtros->w; a++) {
		fread(data, 1, c->filtros->bytes, src);
		error->error = TensorPutValuesOffSet(queue, c->filtros, data, a * c->filtros->bytes);
		if (error->error) {
			getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
			break;
		}
	}
	free(data);
	return (Camada) c;
}

void releaseConv(CamadaConv *pc) {
	CamadaConv c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->filtros);
	releaseTensor(&c->grad_filtros);
	releaseKernel(&c->kernelConvFixWeight);
	releaseKernel(&c->kernelConvSum);
	releaseKernel(&c->kernelConvCalcGrads);
	free(c);
	*pc = NULL;
}

Camada createConv(WrapperCL *cl, QUEUE queue, UINT passo, UINT lenFilter,
                  UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params params, char usehost, Exception *error, int randomize) {
	if (error->error)return NULL;
	CamadaConv c = (CamadaConv) calloc(1, sizeof(Typecamadaconv));
	__newCamada__(&c->super, cl, CONV, entrada, queue, params,
	              inx, iny, inz,
	              (inx - lenFilter) / passo + 1, (iny - lenFilter) / passo + 1,
	              numeroFiltros, usehost, error);

	c->super.toString = (cfv) tostringConv;
	c->super.getCreateParams = (cfv) getCreateParamsConv;

	c->super.release = (fv) releaseConv;
	c->super.ativa = (fv) ativaConv;
	c->super.calc_grads = (f2v) calc_gradsConv;
	c->super.corrige_pesos = (fv) corrige_pesosConv;
	c->super.salvar = (f4v) salvarConv;
	c->passo = passo;
	if (error->error) {
		c->super.release(&c);
		return NULL;

	}
	c->filtros = newTensor4D(cl->context, queue, lenFilter, lenFilter, inz, numeroFiltros, c->super.flag_usehost,
	                         error);
	c->grad_filtros = newTensor4D(cl->context, queue, lenFilter, lenFilter, inz, numeroFiltros, c->super.flag_usehost,
	                              error);
	if (error->error) {
		c->super.release(&c);
		return NULL;
	}

	error->error = TensorFill(queue, c->grad_filtros, 0);
	if (error->error) {
		c->super.release(&c);
		return NULL;
	}


	if (randomize) convRandomize(c, cl, error);
	if (error->error) {
		c->super.release(&c);
		return NULL;
	}


	c->kernelConvSum = new_Kernel(cl->program, error, convSum, 13,
	                              K_VOID_P, K_VOID_P, K_VOID_P,
	                              K_INT, K_INT, K_INT, K_INT, K_INT,
	                              K_INT, K_INT, K_INT, K_INT, K_INT);

	c->kernelConvFixWeight = new_Kernel(cl->program, error, convCalcGradAndFixWeight, 17,
	                                    K_VOID_P, K_VOID_P,
	                                    K_VOID_P, K_VOID_P,
	                                    K_INT, K_INT, K_INT,
	                                    K_INT, K_INT,
	                                    K_INT, K_INT,
	                                    K_INT, K_INT,
	                                    K_DOUBLE, K_DOUBLE, K_DOUBLE,
	                                    K_INT
	);

	c->kernelConvCalcGrads = new_Kernel(cl->program, error, convCalcGradIn, 14,
	                                    K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_INT, K_INT, K_INT,
	                                    K_INT, K_INT,
	                                    K_INT, K_INT,
	                                    K_INT, K_INT, K_INT,
	                                    K_INT);
	if (error->error) {
		c->super.release(&c);
		return NULL;
	}

	return (Camada) c;
}
