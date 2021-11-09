//
// Created by Henrique on 5/8/2021.
//
#include "camadas/CamadaRelu.h"

#if (RUN_KERNEL_USING_GPU != 1)
#include "../../kernels/camadas/utils.h"
#include "../../kernels/camadas/relu.h"
#endif

const char *getCreateParamsRelu(CamadaRelu self) {
	if (self->super.__string__ != NULL)free_mem(self->super.__string__);

	self->super.__string__ = mprintf("['Relu',%g,%g]",self->lessoh,self->greateroh);
	return self->super.__string__;
}

const char *tostringRelu(CamadaRelu self) {
	if (self->super.__string__ != NULL)free_mem(self->super.__string__);

	self->super.__string__ = mprintf("Relu  Layer: [%g,%g](%u,%u,%u) -> (%u,%u,%u)\n", self->lessoh,self->greateroh,
									 self->super.entrada->x, self->super.entrada->y,self->super.entrada->z,
					   self->super.saida->x, self->super.saida->y, self->super.saida->z
	);

	return self->super.__string__;
}

void realeaseRelu(CamadaRelu *pc) {
	CamadaRelu c = *pc;
	__releaseCamada__((Camada) c);
	releaseKernel(&c->kernelReluCalcGrads);
	releaseKernel(&c->kernelReluAtiva);
	free_mem(c);
	*pc = NULL;
}

int ativaRelu(CamadaRelu c) {
	int erro = 0;
	kernel_run_recursive(erro, c->kernelReluAtiva, c->super.queue,
						 c->super.saida->x * c->super.saida->y * c->super.saida->z,
						 *c->super.max_works,
						 K_ARG c->super.entrada->data,
						 K_ARG c->super.saida->data,
						 K_ARG c->lessoh,
						 K_ARG c->greateroh);

	return erro;
}


int calc_gradsRelu(CamadaRelu c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->kernelReluCalcGrads, c->super.queue,
						 c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
						 *c->super.max_works,
						 K_ARG c->super.gradsEntrada->data,
						 K_ARG c->super.entrada->data,
						 K_ARG GradNext->data,
						 K_ARG c->lessoh,
						 K_ARG c->greateroh
	);
	return erro;

}

void salvarRelu(WrapperCL *cl, CamadaRelu c, FILE *dst, CNN_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->lessoh, sizeof(REAL), 1, dst);
	fwrite(&c->greateroh, sizeof(REAL), 1, dst);

}

Camada carregarRelu(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada, CNN_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	REAL less = 0, greater = 1;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&less, sizeof(REAL), 1, src);
	fread(&greater, sizeof(REAL), 1, src);
	return createRelu(cl, queue, inx, iny, inz, less, greater, entrada, error);
}

Camada createRelu(WrapperCL *cl, cl_command_queue queue, unsigned int inx, unsigned int iny,
				  unsigned int inz, REAL lessoh, REAL greateroh, Tensor entrada, CNN_ERROR *error) {
	if (error->error)return NULL;

	CamadaRelu self = (CamadaRelu) alloc_mem(1, sizeof(TypecamadaRelu));

	__newCamada__((Camada) self, cl, RELU, entrada, queue, (Params) {0}, inx, iny, inz, inx, iny, inz, error);
	self->greateroh = greateroh;
	self->lessoh = lessoh;
	self->super.toString = (cfv) tostringRelu;
	self->super.getCreateParams = (cfv) getCreateParamsRelu;
	self->super.release = (fv) realeaseRelu;
	self->super.propagation = (fv) ativaRelu;
	self->super.backpropagation = (f2v) calc_gradsRelu;
	self->super.salvar = (f4v) salvarRelu;

	self->kernelReluAtiva = new_Kernel(cl->program, error, reluativa, 3, K_VOID_P, K_VOID_P,K_REAL,K_REAL, K_INT);
	self->kernelReluCalcGrads = new_Kernel(cl->program, error, relucalcgrad, 4, K_VOID_P, K_VOID_P, K_VOID_P,K_REAL,K_REAL, K_INT);
	return (Camada) self;
}
