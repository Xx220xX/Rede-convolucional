//
// Created by Henrique on 5/8/2021.
//

#include "../CamadaConv.h"

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
	                   c->tamanhoFiltro, c->tamanhoFiltro,
	                   c->numeroFiltros
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}


Camada createConv(WrapperCL *cl, QUEUE queue, UINT passo, UINT lenFilter,
                  UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params params, GPU_ERROR *error, int randomize) {
	if (error->error)return NULL;
	CamadaConv c = (CamadaConv) calloc(1, sizeof(Typecamadaconv));
	cl_context context = cl->context;

	__newCamada__(&c->super, cl, CONV, entrada, queue, params,
	              inx, iny, inz,
	              (inx - lenFilter) / passo + 1, (iny - lenFilter) / passo + 1, numeroFiltros, error);
	c->super.toString = (fch) tostringConv;

	c->super.release = (fv) releaseConv;
	c->super.ativa = (fv) ativaConv;
	c->super.calc_grads = (fvv) calc_gradsConv;
	c->super.corrige_pesos = (fv) corrige_pesosConv;
	c->super.salvar = (fsl) salvarConv;
	c->passo = passo;
	c->tamanhoFiltro = lenFilter;
	c->numeroFiltros = numeroFiltros;
	if (error->error)return (Camada) c;
	if (error->error)return (Camada) c;
	c->filtros = newTensor4D(cl->context, lenFilter, lenFilter, inz, numeroFiltros, error);
	c->grad_filtros = newTensor4D(cl->context, lenFilter, lenFilter, inz, numeroFiltros, error);
	c->grad_filtros_old = newTensor4D(cl->context, lenFilter, lenFilter, inz, numeroFiltros, error);


	if (randomize) convRandomize(c, cl, error);
	if (error->error) return (Camada) c;
	c->kernelConvSum = new_Kernel(cl->program,error, "convSum", 11, K_VOID_P, K_VOID_P, K_VOID_P,
	                              K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	c->kernelConvFixWeight = new_Kernel(cl->program,error, "convFixWeight", 7, K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_DOUBLE, K_DOUBLE, K_DOUBLE, K_INT);
	c->kernelConvCalcGradsFiltro = new_Kernel(cl->program,error, "convCalcFiltro", 12,
	                                          K_VOID_P, K_VOID_P, K_VOID_P,
	                                          K_INT, K_INT, K_INT,
	                                          K_INT, K_INT, K_INT,
	                                          K_INT, K_INT, K_INT
	);
	c->kernelConvCalcGrads = new_Kernel(cl->program,error, "convCalcGrads", 13,
	                                    K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_INT, K_INT, K_INT,
	                                    K_INT, K_INT, K_INT,
	                                    K_INT, K_INT, K_INT);
	return (Camada) c;
}

int convRandomize(CamadaConv c, WrapperCL *cl, GPU_ERROR *error) {
	int lenFilter = c->tamanhoFiltro,
			inz = c->super.entrada->z,
			numeroFiltros = c->numeroFiltros;
	cl_context context = cl->context;
	double maxVal = 1.0 / (double) (lenFilter * lenFilter * inz);

	double *data = (double *) calloc(lenFilter * lenFilter * inz, sizeof(double));
	QUEUE queue = c->super.queue;
	if (error->error) {
		snprintf(error->msg, 255, "nao foi possivel cruiar a queue\n");
		return error->error;
	}
	for (int a = 0; a < numeroFiltros; a++) {
		FOR3D(i, j, z, lenFilter, lenFilter, inz) {
					data[TensorMap(c->filtros, i, j, z)] = RANDOM_BILATERAL() * maxVal;
				}
		error->error = clEnqueueWriteBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes,
		                                    data, 0, NULL, NULL);
		clFinish(queue);
		if (error->error) {
			fprintf(stderr, "warning inside randomize conv\n");
			snprintf(error->msg, 255, "nao foi possivel copiar dados\n");
			free(data);
			return error->error;

		}
	}
	free(data);
	return 0;
}

void releaseConv(CamadaConv *pc) {
	CamadaConv c = *pc;
	releaseTensor(&c->super.gradsEntrada);
	if (c->super.flag_releaseInput)
		releaseTensor(&c->super.entrada);
	releaseTensor(&c->super.saida);
	if (c->super.__string__ != NULL) {
		free(c->super.__string__);
	}
	releaseTensor(&c->filtros);
	releaseTensor(&c->grad_filtros);
	releaseTensor(&c->grad_filtros_old);
	releaseKernel(&c->kernelConvFixWeight);
	releaseKernel(&c->kernelConvSum);
	releaseKernel(&c->kernelConvCalcGradsFiltro);
	releaseKernel(&c->kernelConvCalcGrads);

	free(c);
	*pc = NULL;
}

int ativaConv(CamadaConv c) {
	//iteraçao nos filtros
	kernel_run_recursive(&c->kernelConvSum, c->super.queue, c->super.saida->x * c->super.saida->y * c->numeroFiltros,
	                     *c->super.max_works,
	                     &c->filtros->data, &c->super.entrada->data, &c->super.saida->data,
	                     &c->passo, &c->super.saida->x, &c->super.saida->y, &c->super.entrada->x, &c->super.entrada->y,
	                     &c->tamanhoFiltro, &c->super.entrada->z);
	return 0;
}

void corrige_pesosConv(CamadaConv c) {
	kernel_run_recursive(&c->kernelConvFixWeight, c->super.queue,
	                     c->tamanhoFiltro * c->tamanhoFiltro * c->super.entrada->z * c->numeroFiltros,
	                     *c->super.max_works,
	                     &c->filtros->data,
	                     &c->grad_filtros->data,
	                     &c->grad_filtros_old->data,
	                     &c->super.parametros.hitLearn,
	                     &c->super.parametros.momento,
	                     &c->super.parametros.decaimentoDePeso);
}

void calc_gradsConv(CamadaConv c, Tensor Gradnext) {
	kernel_run_recursive(&c->kernelConvCalcGradsFiltro, c->super.queue,
	                     c->filtros->x * c->filtros->y * c->filtros->z * c->numeroFiltros,
	                     *c->super.max_works,
	                     &Gradnext->data,
	                     &c->super.entrada->data,
	                     &c->grad_filtros->data,
	                     &c->filtros->x,
	                     &c->filtros->y,
	                     &c->filtros->z,
	                     &c->super.entrada->x,
	                     &c->super.entrada->y,
	                     &c->super.saida->x,
	                     &c->super.saida->y,
	                     &c->passo
	);
	//clFinish(c->super.queue);

	kernel_run_recursive(&c->kernelConvCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->filtros->data,
	                     &c->super.entrada->data,
	                     &c->super.gradsEntrada->data,
	                     &Gradnext->data,
	                     &c->tamanhoFiltro,
	                     &c->filtros->z,
	                     &c->passo,
	                     &c->super.entrada->x,
	                     &c->super.entrada->y,
	                     &c->super.saida->x,
	                     &c->super.saida->y,
	                     &c->numeroFiltros);


}

void salvarConv(WrapperCL *cl, CamadaConv c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->passo, sizeof(UINT), 1, dst);
	fwrite(&c->tamanhoFiltro, sizeof(UINT), 1, dst);
	fwrite(&c->numeroFiltros, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	double *data = callocdouble(c->tamanhoFiltro * c->tamanhoFiltro * c->super.entrada->z);
	QUEUE queue = c->super.queue;
	for (int a = 0; a < c->numeroFiltros; a++) {
		clEnqueueReadBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes, data, 0, NULL,
		                    NULL);
		fwrite(data, 1, c->filtros->bytes, dst);
	}
	clFinish(queue);
}

Camada carregarConv(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada,
                    Params params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT passo, tamanhoFiltro, numeroFiltros, inx, iny, inz;
	fread(&passo, sizeof(UINT), 1, src);
	fread(&tamanhoFiltro, sizeof(UINT), 1, src);
	fread(&numeroFiltros, sizeof(UINT), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	CamadaConv c = (CamadaConv) createConv(cl, queue, passo, tamanhoFiltro, numeroFiltros, inx, iny, inz, entrada,
	                                       params,
	                                       error, 0);
	double *data = callocdouble(c->tamanhoFiltro * c->tamanhoFiltro * c->super.entrada->z);
	for (int a = 0; a < c->numeroFiltros; a++) {
		fread(data, 1, c->filtros->bytes, src);
		clEnqueueWriteBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes, data, 0, NULL,
		                     NULL);
	}
	clFinish(queue);
	return (Camada) c;
}
