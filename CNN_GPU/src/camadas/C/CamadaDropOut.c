//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaDropOut.h"

const char *tostringDropOut(CamadaDropOut c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Drop Out Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\tactivation point  %lf\n",

	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->p_ativacao

	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

Camada createDropOut(WrapperCL *cl, cl_command_queue queue, UINT inx, UINT iny, UINT inz,
                     double p_ativacao, long long seed, Tensor entrada,
                     GPU_ERROR *error) {
	if (error->error)return NULL;
	CamadaDropOut c = (CamadaDropOut) calloc(1, sizeof(Typecamadadropout));
	__newCamada__((Camada) c, cl, DROPOUT, entrada, queue, (Params) {0}, inx, iny, inz, inx, iny, inz, error);
	c->super.toString = (fch) tostringDropOut;
	c->hitmap = newTensorChar(cl->context, inx, iny, inz, error);
	c->p_ativacao = p_ativacao;
	c->super.release = (fv) releaseDropOut;
	c->super.ativa = (fv) ativaDropOut;
	c->super.calc_grads = (fvv) calc_gradsDropOut;
	c->super.corrige_pesos = (fv) corrigePesosDropOut;
	c->seed = seed;
	c->super.salvar = (fsl) salvarDropOut;
	c->kerneldropativa = new_Kernel(cl->program,error, "dropativa", 6, K_VOID_P, K_VOID_P, K_VOID_P, sizeof(cl_long),
	                                K_DOUBLE, K_INT);
	c->kerneldropcalcgrad = new_Kernel(cl->program,error, "dropcalcgrad", 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	return (Camada) c;
}

void releaseDropOut(CamadaDropOut *pc) {
	CamadaDropOut c = *pc;
	releaseTensor(&c->super.gradsEntrada);
	releaseTensor(&c->super.saida);
	releaseTensorChar(&c->hitmap);
	if (c->super.__string__ != NULL) {
		free(c->super.__string__);
	}
	if (c->flag_releaseInput)releaseTensor(&c->super.entrada);
	releaseKernel(&c->kerneldropcalcgrad);
	releaseKernel(&c->kerneldropativa);
	free(c);
	*pc = 0;
}

void ativaDropOut(CamadaDropOut c) {
	kernel_run_recursive(&c->kerneldropativa, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     &c->super.entrada->data, &c->super.saida->data, &c->hitmap->data, &c->seed, &c->p_ativacao
	);
	c->seed += c->super.saida->x * c->super.saida->y * c->super.saida->z;
	c->seed = (c->seed * 0x5deece66dLL + 0xbLL) & ((1LL << 48) - 1);
}

void corrigePesosDropOut(CamadaDropOut c) {}

void calc_gradsDropOut(CamadaDropOut c, Tensor GradNext) {
	kernel_run_recursive(&c->kerneldropcalcgrad, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works, &c->super.gradsEntrada->data,
	                     &c->hitmap->data, &GradNext->data);

}

void salvarDropOut(WrapperCL *cl, CamadaDropOut c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->p_ativacao, sizeof(double), 1, dst);

}

Camada carregarDropOut(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                       Params params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	double pativacao;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&pativacao, sizeof(double), 1, src);
	CamadaDropOut c = (CamadaDropOut) createDropOut(cl, queue, inx, iny, inz, pativacao,
												 time(NULL), entrada, error);
	return (Camada) c;
}


