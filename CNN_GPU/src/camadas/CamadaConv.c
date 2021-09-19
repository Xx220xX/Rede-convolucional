//
// Created by Henrique on 5/8/2021.
//

#include "camadas/CamadaConv.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/conv.h"
#endif


const char *getCreateParamsConv(CamadaConv c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "['Convolucao',%d,%d,%d,%d,%d]",
					   c->passox,
					   c->passoy,
					   c->filtros->x,
					   c->filtros->y,
					   c->filtros->w
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringConv(CamadaConv c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
					   "Convolutional Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
					   "\tstep %u %u\n"
					   "\tfilter dim (%u %u)\n"
					   "\tnumber of filters %u\n",

					   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
					   c->super.saida->x, c->super.saida->y, c->super.saida->z,
					   c->passox, c->passoy,
					   c->filtros->x, c->filtros->y,
					   c->filtros->w
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}


int ativaConv(CamadaConv c) {
	//iteraçao nos filtros
	int erro = 0;


	kernel_run_recursive(erro, c->kernelConvSum, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->filtros->data, K_ARG c->super.entrada->data,
						 K_ARG c->super.saida->data,
						 K_ARG c->passox, K_ARG c->passoy,
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
						 K_ARG c->filtros->data,
						 K_ARG c->gradnext->data,
						 K_ARG c->super.entrada->data,
						 K_ARG c->grad_filtros->data,
						 K_ARG c->filtros->x, K_ARG c->filtros->y, K_ARG c->filtros->z,
						 K_ARG c->super.entrada->x, K_ARG c->super.entrada->y,
						 K_ARG c->super.saida->x, K_ARG c->super.saida->y,
						 K_ARG c->passox, K_ARG c->passoy,
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
						 K_ARG c->filtros->data,
						 K_ARG c->super.gradsEntrada->data,
						 K_ARG Gradnext->data,
						 K_ARG c->filtros->x, K_ARG c->filtros->y, K_ARG c->filtros->z,
						 K_ARG c->passox, K_ARG c->passoy,
						 K_ARG c->super.entrada->x, K_ARG c->super.entrada->y,
						 K_ARG c->super.saida->x, K_ARG c->super.saida->y, K_ARG c->super.saida->z);

	if (erro)return erro;
	if (c->super.learnable)return corrige_pesosConv(c);
	return erro;


}

void salvarConv(WrapperCL *cl, CamadaConv c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);

	fwrite(&c->passox, sizeof(UINT), 1, dst);
	fwrite(&c->passoy, sizeof(UINT), 1, dst);
	fwrite(&c->filtros->x, sizeof(UINT), 1, dst);
	fwrite(&c->filtros->y, sizeof(UINT), 1, dst);
	fwrite(&c->filtros->w, sizeof(UINT), 1, dst);

	double *data = (double *) alloc_mem(c->filtros->x * c->filtros->y * c->filtros->z * c->filtros->w, sizeof(double));
	TensorGetValuesMem(c->super.queue, c->filtros, data, c->filtros->bytes * c->filtros->w);
	fwrite(data, 1, c->filtros->bytes * c->filtros->w, dst);
	free_mem(data);
}

Camada carregarConv(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada,
					Params params, CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	UINT fx, fy, fw, px, py, inx, iny, inz;

	fread(&flag, sizeof(char), 1, src);
	if (flag != '#') {
		fread(&flag, sizeof(char), 1, src);
	}
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);

	fread(&px, sizeof(UINT), 1, src);
	fread(&py, sizeof(UINT), 1, src);
	fread(&fx, sizeof(UINT), 1, src);
	fread(&fy, sizeof(UINT), 1, src);
	fread(&fw, sizeof(UINT), 1, src);
	CamadaConv c = (CamadaConv) createConv(cl, queue, px, py, fx, fy, fw, inx, iny, inz, entrada,params, (RandomParam) {-1}, error);
	if(error->error){
		c->super.release(c);
		return NULL;
	}
	double *data = (double *) alloc_mem(fx * fy * inz * fw, sizeof(double));
	fread(data, 1, fx * fy * inz * fw * sizeof(double), src);
	error->error = TensorPutValuesMem(queue, c->filtros, data, c->filtros->bytes * c->filtros->w);
	free_mem(data);
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
	free_mem(c);
	*pc = NULL;
}

Camada createConv(WrapperCL *cl, QUEUE queue, UINT passox, UINT passoy, UINT lenFilterx, UINT lenFiltery,
				  UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
				  Tensor entrada, Params params, RandomParam randomParams, CNN_ERROR *error) {
	if (error->error)return NULL;
	CamadaConv c = (CamadaConv) alloc_mem(1, sizeof(Typecamadaconv));
	__newCamada__(&c->super, cl, CONV, entrada, queue, params,
				  inx, iny, inz,
				  (inx - lenFilterx) / passox + 1, (iny - lenFiltery) / passoy + 1,
				  numeroFiltros, error);

	c->super.toString = (cfv) tostringConv;
	c->super.getCreateParams = (cfv) getCreateParamsConv;

	c->super.release = (fv) releaseConv;
	c->super.propagation = (fv) ativaConv;
	c->super.backpropagation = (f2v) calc_gradsConv;
	c->super.salvar = (f4v) salvarConv;
	c->passox = passox;
	c->passoy = passoy;
	if (error->error) {
		c->super.release(&c);
		return NULL;

	}
	c->filtros = newTensor4D(cl->context, queue, lenFilterx, lenFiltery, inz, numeroFiltros, 0,
							 error);
	c->grad_filtros = newTensor4D(cl->context, queue, lenFilterx, lenFiltery, inz, numeroFiltros, 0,
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


	if (randomParams.type != -1) {
		if (randomParams.type == 0)
			TensorRandomize(queue, c->filtros, LCG_UNIFORM,
							2.0 / (c->filtros->x * c->filtros->y * c->filtros->z),
							-1.0 / (c->filtros->x * c->filtros->y * c->filtros->z));
		else
			TensorRandomize(queue, c->filtros, randomParams.type,
							randomParams.a,
							randomParams.b);
	}
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
