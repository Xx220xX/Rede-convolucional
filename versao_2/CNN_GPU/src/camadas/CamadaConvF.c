//
// Created by Henrique on 5/8/2021.
//

#include "camadas/CamadaConvF.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../kernels/camadas/utils.h"
#include "../../kernels/camadas/convf.h"
#endif


const char *getCreateParamsConvF(CamadaConvF c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = mprintf(
			"['ConvolucaoF',%d,%d,%d,%d,%d,%d]",
			c->passox,
			c->passoy,
			c->filtros->x,
			c->filtros->y,
			c->filtros->w,
			c->activationFuntion
	);
	return c->super.__string__;
}

const char *tostringConvF(CamadaConvF c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = mprintf(
			"Convolutional activation Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
			"\tstep %u %u\n"
			"\tfilter dim (%u %u)\n"
			"\tnumber of filters %u\n"
			"\tactivation %u\n"
			,

			c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
			c->super.saida->x, c->super.saida->y, c->super.saida->z,
			c->passox, c->passoy,
			c->filtros->x, c->filtros->y,
			c->filtros->w,c->activationFuntion
	);
	return c->super.__string__;
}


int ativaConvF(CamadaConvF self) {
	//iteraçao nos filtros
	int erro = 0;


	kernel_run_recursive(erro, self->kernelConvFSum, self->super.queue,
						 self->super.saida->x * self->super.saida->y * self->super.saida->z,
						 *self->super.max_works,
						 K_ARG self->filtros->data, K_ARG self->super.entrada->data,
						 K_ARG self->z->data, K_ARG self->super.saida->data,
						 K_ARG self->passox, K_ARG self->passoy,
						 K_ARG self->super.saida->x, K_ARG self->super.saida->y,
						 K_ARG self->super.entrada->x, K_ARG self->super.entrada->y,
						 K_ARG self->filtros->x, K_ARG self->filtros->y, K_ARG self->filtros->z,
						 K_ARG self->activationFuntion);
	return erro;
}

int corrige_pesosConvF(CamadaConvF c, Tensor Gradnext) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelConvFFixWeight, c->super.queue,
						 c->filtros->x * c->filtros->y * c->filtros->z * c->filtros->w,
						 *c->super.max_works,
						 K_ARG c->filtros->data,
						 K_ARG Gradnext->data,
						 K_ARG c->super.entrada->data,
						 K_ARG c->grad_filtros->data,
						 K_ARG c->filtros->x, K_ARG c->filtros->y, K_ARG c->filtros->z,
						 K_ARG c->super.entrada->x, K_ARG c->super.entrada->y,
						 K_ARG c->super.saida->x, K_ARG c->super.saida->y,
						 K_ARG c->passox, K_ARG c->passoy,
						 K_ARG c->super.parametros.hitLearn,
						 K_ARG c->super.parametros.momento,
						 K_ARG c->super.parametros.decaimentoDePeso,
						 K_ARG c->derivationFuntion);
	return erro;
}

int calc_gradsConvF(CamadaConvF self, Tensor Gradnext) {

	int erro = 0;

// calcula gradiente da saida
	kernel_run_recursive(erro, self->kernelConvFCalcZGrad, self->super.queue,
						 self->super.saida->x * self->super.saida->y * self->super.saida->z,
						 *self->super.max_works,
						 K_ARG Gradnext->data,
						 K_ARG self->z->data,
						 K_ARG self->dz->data,
						 K_ARG self->derivationFuntion
	);
	if (erro)return erro;
// calcula gradiente de entrada
	if (self->super.gradsEntrada) {
		kernel_run_recursive(erro, self->kernelConvFCalcGrads, self->super.queue,
							 self->super.entrada->x * self->super.entrada->y * self->super.entrada->z,
							 *self->super.max_works,
							 K_ARG self->filtros->data,
							 K_ARG self->super.gradsEntrada->data,
							 K_ARG Gradnext->data,
							 K_ARG self->filtros->x, K_ARG self->filtros->y, K_ARG self->filtros->z,
							 K_ARG self->passox, K_ARG self->passoy,
							 K_ARG self->super.entrada->x, K_ARG self->super.entrada->y,
							 K_ARG self->super.saida->x, K_ARG self->super.saida->y, K_ARG self->super.saida->z, K_ARG self->derivationFuntion);
		if (erro)return erro;
	}
	//calcular gradiente interno e corrige peso
	if (self->super.learnable)
		return corrige_pesosConvF(self, Gradnext);
	return erro;


}

void salvarConvF(WrapperCL *cl, CamadaConvF c, FILE *dst, CNN_ERROR *error) {
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
	fwrite(&c->super.parametros, sizeof(Params), 1, dst);
	fwrite(&c->activationFuntion, sizeof(int), 1, dst);

	REAL *data = (REAL *) alloc_mem(c->filtros->x * c->filtros->y * c->filtros->z * c->filtros->w, sizeof(REAL));
	TensorGetValuesMem(c->super.queue, c->filtros, data, c->filtros->bytes * c->filtros->w);
	fwrite(data, 1, c->filtros->bytes * c->filtros->w, dst);
	free_mem(data);
}

Camada carregarConvF(WrapperCL *cl, FILE *src, QUEUE queue, Tensor entrada,
					 CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	UINT fx, fy, fw, px, py, inx, iny, inz;
	int ativa = 2;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#') {
		fread(&flag, sizeof(char), 1, src);
	}
	Params params;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);

	fread(&px, sizeof(UINT), 1, src);
	fread(&py, sizeof(UINT), 1, src);
	fread(&fx, sizeof(UINT), 1, src);
	fread(&fy, sizeof(UINT), 1, src);
	fread(&fw, sizeof(UINT), 1, src);
	fread(&params, sizeof(Params), 1, src);
	fread(&ativa, sizeof(int), 1, src);

	CamadaConvF c = (CamadaConvF) createConvF(cl, queue, px, py, fx, fy, fw, inx, iny, inz, ativa, entrada, params, (RandomParam) {-1}, error);
	if (error->error) {
		c->super.release(c);
		return NULL;
	}
	REAL *data = (REAL *) alloc_mem(fx * fy * inz * fw, sizeof(REAL));
	fread(data, 1, fx * fy * inz * fw * sizeof(REAL), src);
	error->error = TensorPutValuesMem(queue, c->filtros, data, c->filtros->bytes * c->filtros->w);
	free_mem(data);
	return (Camada) c;
}

void releaseConvF(CamadaConvF *pc) {
	CamadaConvF c = *pc;
	__releaseCamada__((Camada) c);
	releaseTensor(&c->filtros);
	releaseTensor(&c->grad_filtros);
	releaseTensor(&c->z);
	releaseTensor(&c->dz);
	releaseKernel(&c->kernelConvFFixWeight);
	releaseKernel(&c->kernelConvFSum);
	releaseKernel(&c->kernelConvFCalcGrads);
	releaseKernel(&c->kernelConvFCalcZGrad);
	free_mem(c);
	*pc = NULL;
}

Camada createConvF(WrapperCL *cl, QUEUE queue, UINT passox, UINT passoy, UINT lenFilterx, UINT lenFiltery,
				   UINT numeroFiltros, UINT inx, UINT iny, UINT inz, int ativacao,
				   Tensor entrada, Params params, RandomParam randomParams, CNN_ERROR *error) {
	if (error->error)return NULL;
	CamadaConvF self = (CamadaConvF) alloc_mem(1, sizeof(Typecamadaconvf));
	__newCamada__(&self->super, cl, CONV, entrada, queue, params,
				  inx, iny, inz,
				  (inx - lenFilterx) / passox + 1, (iny - lenFiltery) / passoy + 1,
				  numeroFiltros, error);

	self->super.toString = (cfv) tostringConvF;
	self->super.getCreateParams = (cfv) getCreateParamsConvF;

	self->super.release = (fv) releaseConvF;
	self->super.propagation = (fv) ativaConvF;
	self->super.backpropagation = (f2v) calc_gradsConvF;
	self->super.salvar = (f4v) salvarConvF;
	self->passox = passox;
	self->passoy = passoy;
	self->activationFuntion = ativacao;
	self->derivationFuntion = ativacao | FLAGDIF;
	if (error->error) {
		self->super.release(&self);
		return NULL;

	}
	self->filtros = newTensor4D(cl->context, queue, lenFilterx, lenFiltery, inz, numeroFiltros, 0,
								error);
	self->grad_filtros = newTensor4D(cl->context, queue, lenFilterx, lenFiltery, inz, numeroFiltros, 0,
									 error);
	self->z = newTensor(cl->context, queue, self->super.saida->x, self->super.saida->y, self->super.saida->z, 0, error);
	self->dz = newTensor(cl->context, queue, self->super.saida->x, self->super.saida->y, self->super.saida->z, 0, error);
	if (error->error) {
		self->super.release(&self);
		return NULL;
	}

	error->error = TensorFill(queue, self->grad_filtros, 0);
	if (error->error) {
		self->super.release(&self);
		return NULL;
	}


	if (randomParams.type != -1) {
		if (randomParams.type == 0)
			TensorRandomize(queue, self->filtros, LCG_UNIFORM,
							2.0 / (self->filtros->x * self->filtros->y * self->filtros->z),
							-1.0 / (self->filtros->x * self->filtros->y * self->filtros->z));
		else
			TensorRandomize(queue, self->filtros, randomParams.type,
							randomParams.a,
							randomParams.b);
	}
	if (error->error) {
		self->super.release(&self);
		return NULL;
	}


	self->kernelConvFSum = new_Kernel(cl->program, error, convFSum, 15,
									  K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
									  K_INT, K_INT, K_INT, K_INT, K_INT,K_INT,
									  K_INT, K_INT, K_INT,K_INT, K_INT);

	self->kernelConvFFixWeight = new_Kernel(cl->program, error, convFCalcGradAndFixWeight, 17,
											K_VOID_P, K_VOID_P,
											K_VOID_P, K_VOID_P,
											K_INT, K_INT, K_INT,
											K_INT, K_INT,
											K_INT, K_INT,
											K_INT, K_INT,
											K_REAL, K_REAL, K_REAL,
											K_INT
	);

	self->kernelConvFCalcGrads = new_Kernel(cl->program, error, convFCalcGradIn, 14,
											K_VOID_P, K_VOID_P, K_VOID_P,
											K_INT, K_INT, K_INT,
											K_INT, K_INT,
											K_INT, K_INT,
											K_INT, K_INT, K_INT,
											K_INT);
	self->kernelConvFCalcZGrad = new_Kernel(cl->program, error, convFCalcGradZ, 5,
											K_VOID_P, K_VOID_P, K_VOID_P,
											K_INT, K_INT);
	if (error->error) {
		self->super.release(&self);
		return NULL;
	}

	return (Camada) self;
}
