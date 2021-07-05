//
// Created by Henrique on 05-jul-2021.
//

#include "CamadaConvNc.h"

const char *tostringConvNc(CamadaConvNc c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Convolutional Non-Causal Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\tstep %u %u\n"
	                   "\tlagura %u %u\n"
	                   "\tfilter dim (%u %u)\n"
	                   "\tnumber of filters %u\n",

	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->passox, c->passoy,
	                   c->largx, c->passoy,
	                   c->filtros->x, c->filtros->y,
	                   c->numeroFiltros
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

Camada createConvNc(WrapperCL *cl, cl_command_queue queue, UINT passox,
                    UINT passoy, UINT largx, UINT largy, UINT filtrox, UINT filtroy,
                    UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                    Tensor entrada, Params *params, GPU_ERROR *error, int randomize) {
	if (error->error)return NULL;
	CamadaConvNc c = (CamadaConvNc) calloc(1, sizeof(TypecamadaConvNc));


	__newCamada__(&c->super, cl, CONVNC, entrada, queue, params,
	              inx, iny, inz,
	              (inx - (filtrox - 1) * largx) / passox,
	              (iny - (filtroy - 1) * largy) / passoy,
	              numeroFiltros, error);

	c->super.toString = (fch) tostringConvNc;
	c->super.release = (fv) releaseConvNc;
	c->super.ativa = (fv) ativaConvNc;
	c->super.calc_grads = (fvv) calc_gradsConvNc;
	c->super.corrige_pesos = (fv) corrige_pesosConvNc;
	c->super.salvar = (fsl) salvarConvNc;
	c->passox = passox;
	c->passoy = passoy;
	c->largx = largx;
	c->largy = largy;
	c->numeroFiltros = numeroFiltros;
	if (error->error)return (Camada) c;
	if (error->error)return (Camada) c;
	c->filtros = newTensor4D(cl->context, filtrox, filtroy, inz, numeroFiltros, error);
	c->grad_filtros = newTensor4D(cl->context, filtrox, filtroy, inz, numeroFiltros, error);
	c->grad_filtros_old = newTensor4D(cl->context, filtrox, filtroy, inz, numeroFiltros, error);


	if (randomize) ConvNcRandomize(c, cl, error);
	if (error->error) return (Camada) c;
	c->kernelConvNcSum = new_Kernel(cl->program, "convncSum", 15,
	                                K_VOID_P, K_VOID_P, K_VOID_P,
	                                K_INT, K_INT, K_INT,
	                                K_INT, K_INT, K_INT,
	                                K_INT, K_INT, K_INT, K_INT,
	                                K_INT, K_INT);
	c->kernelConvNcFixWeight = new_Kernel(cl->program, "convncFixWeight", 7, K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_DOUBLE, K_DOUBLE, K_DOUBLE, K_INT);
	c->kernelConvNcCalcGradsFiltro = new_Kernel(cl->program, "ConvNcCalcFiltro", 12,
	                                            K_VOID_P, K_VOID_P, K_VOID_P,
	                                            K_INT, K_INT, K_INT,
	                                            K_INT, K_INT, K_INT,
	                                            K_INT, K_INT, K_INT
	);
	c->kernelConvNcCalcGrads = new_Kernel(cl->program, "convncCalcFiltro",
	                                      15,
	                                      K_VOID_P, K_VOID_P, K_VOID_P,
	                                      K_INT, K_INT, K_INT,
	                                      K_INT, K_INT,
	                                      K_INT, K_INT,
	                                      K_INT, K_INT,
	                                      K_INT, K_INT,
	                                      K_INT);
	return (Camada) c;
}

int ConvNcRandomize(CamadaConvNc c, WrapperCL *cl, GPU_ERROR *error) {
	UINT fx = c->filtros->x, fy = c->filtros->y,
			inz = c->super.entrada->z,
			numeroFiltros = c->numeroFiltros;
	cl_context context = cl->context;
	double maxVal = 1.0 / (double) (fx * fx * inz);

	double *data = (double *) calloc(fx * fy * inz, sizeof(double));
	cl_command_queue queue = c->super.queue;
	if (error->error) {
		snprintf(error->msg, 255, "nao foi possivel criar a queue\n");
		return error->error;
	}
	for (int a = 0; a < numeroFiltros; a++) {
		FOR3D(i, j, z, fx, fy, inz) {
					data[TensorMap(c->filtros, i, j, z)] = RANDOM_BILATERAL() * maxVal;
				}
		error->error = clEnqueueWriteBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes,
		                                    data, 0, NULL, NULL);
		clFinish(queue);
		if (error->error) {
			fprintf(stderr, "warning inside randomize ConvNc\n");
			snprintf(error->msg, 255, "nao foi possivel copiar dados\n");
			free(data);
			return error->error;

		}
	}
	free(data);
	return 0;
}

void releaseConvNc(CamadaConvNc *pc) {
	CamadaConvNc c = *pc;
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
	releaseKernel(&c->kernelConvNcFixWeight);
	releaseKernel(&c->kernelConvNcSum);
	releaseKernel(&c->kernelConvNcCalcGradsFiltro);
	releaseKernel(&c->kernelConvNcCalcGrads);

	free(c);
	*pc = NULL;
}

int ativaConvNc(CamadaConvNc c) {
	//iteraçao nos filtros
	kernel_run_recursive(&c->kernelConvNcSum, c->super.queue, c->super.saida->x * c->super.saida->y * c->numeroFiltros,
	                     *c->super.max_works,
	                     &c->filtros->data, &c->super.entrada->data, &c->super.saida->data,
	                     &c->passox, &c->passoy, &c->largx,
	                     &c->largy, &c->super.saida->x, &c->super.saida->y,
	                     &c->super.entrada->x, &c->super.entrada->y,
	                     &c->filtros->x, &c->filtros->y,
	                     &c->super.entrada->z);
	return 0;
}

void corrige_pesosConvNc(CamadaConvNc c) {
	kernel_run_recursive(&c->kernelConvNcFixWeight, c->super.queue,
	                     c->filtros->x * c->filtros->y * c->super.entrada->z * c->numeroFiltros,
	                     *c->super.max_works,
	                     &c->filtros->data,
	                     &c->grad_filtros->data,
	                     &c->grad_filtros_old->data,
	                     &c->super.parametros->hitLearn,
	                     &c->super.parametros->momento,
	                     &c->super.parametros->decaimentoDePeso);
}

void calc_gradsConvNc(CamadaConvNc c, Tensor Gradnext) {
	kernel_run_recursive(&c->kernelConvNcCalcGradsFiltro, c->super.queue,
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
	                     &c->passox,
	                     &c->passoy,
	                     &c->largx,
	                     &c->largy

	);

	kernel_run_recursive(&c->kernelConvNcCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->filtros->data,
	                     &c->super.entrada->data,
	                     &c->super.gradsEntrada->data,
	                     &Gradnext->data,

	                     &c->passox,
	                     &c->passoy,
	                     &c->largx,
	                     &c->largy,

	                     &c->super.entrada->x,
	                     &c->super.entrada->y,

	                     &c->super.saida->x,
	                     &c->super.saida->y,
	                     &c->filtros->x,
	                     &c->filtros->y,
	                     &c->filtros->z,
	                     &c->numeroFiltros);

}

void salvarConvNc(WrapperCL *cl, CamadaConvNc c, FILE *dst, GPU_ERROR *error) {
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
	double *data = callocdouble(c->filtros->x * c->filtros->y * c->super.entrada->z);
	cl_command_queue queue = c->super.queue;
	for (int a = 0; a < c->numeroFiltros; a++) {
		clEnqueueReadBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes, data, 0, NULL,
		                    NULL);
		fwrite(data, 1, c->filtros->bytes, dst);
	}
	clFinish(queue);
}

Camada
carregarConvNc(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
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
	CamadaConvNc c = (CamadaConvNc) createConvNc(cl, queue, passox, passoy, largx, largy, fx, fy, numeroFiltros, inx,
	                                             iny, inz,
	                                             entrada, params, error, 0);
	double *data = callocdouble(c->filtros->x * c->filtros->y * c->super.entrada->z);
	for (int a = 0; a < c->numeroFiltros; a++) {
		fread(data, 1, c->filtros->bytes, src);
		clEnqueueWriteBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes, data, 0, NULL,
		                     NULL);
	}
	clFinish(queue);
	return (Camada) c;
}
