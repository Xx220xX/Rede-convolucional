//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaPool.h"

const char *tostringPool(CamadaPool c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Pooling  Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\tStep %u\n",

	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->passo
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

Camada createPool(WrapperCL *cl, cl_command_queue queue, UINT passo, UINT tamanhoFiltro,
                  UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params params,
                  GPU_ERROR *error) {
	CamadaPool c = (CamadaPool) calloc(1, sizeof(Typecamadapool));
	c->passo = passo;
	c->tamanhoFiltro = tamanhoFiltro;
	__newCamada__((Camada) c, cl, POOL, entrada, queue, params, inx, iny, inz, (inx - tamanhoFiltro) / passo + 1,
	              (iny - tamanhoFiltro) / passo + 1, inz,
	              error);
	c->super.toString = (fch) tostringPool;
	c->super.release = (fv) releasePool;
	c->super.ativa = (fv) ativaPool;
	c->super.corrige_pesos = (fv) corrige_pesosPool;
	c->super.calc_grads = (fvv) calc_gradsPool;
	c->super.parametros = params;
	c->super.salvar = (fsl) salvarPool;

	c->kernelPoolAtiva = new_Kernel(cl->program, "poolativa", 9, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT, K_INT, K_INT,
	                                K_INT, K_INT);
	c->kernelPoolCalcGrads = new_Kernel(cl->program, "poolCalcGrads", 12, K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	return (Camada) c;
}

void releasePool(CamadaPool *pc) {
	CamadaPool c = *pc;
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
	if (c->super.__string__ != NULL) {
		free(c->super.__string__);
	}
	releaseTensor(&c->super.gradsEntrada);
	releaseTensor(&c->super.saida);
	releaseKernel(&c->kernelPoolCalcGrads);
	releaseKernel(&c->kernelPoolAtiva);
	free(c);
	*pc = NULL;
}

void ativaPool(CamadaPool c) {
	kernel_run_recursive(&c->kernelPoolAtiva, c->super.queue, c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     &c->super.entrada->data, &c->super.saida->data, &c->tamanhoFiltro, &c->passo,
	                     &c->super.saida->x, &c->super.saida->y, &c->super.entrada->x, &c->super.entrada->y);
}

void corrige_pesosPool(CamadaPool c) {}


void calc_gradsPool(CamadaPool c, Tensor GradNext) {
	kernel_run_recursive(&c->kernelPoolCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->super.entrada->data, &c->super.gradsEntrada->data,
	                     &GradNext->data,
	                     &c->super.saida->data, &c->tamanhoFiltro, &c->passo, &c->super.entrada->x,
	                     &c->super.entrada->y, &c->super.entrada->z,
	                     &c->super.saida->x, &c->super.saida->y);
}

void salvarPool(WrapperCL *cl, CamadaPool c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->passo, sizeof(UINT), 1, dst);
	fwrite(&c->tamanhoFiltro, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
}

Camada carregarPool(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
                    Params params, GPU_ERROR *error) {
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT passo, tamanhoFiltro, inx, iny, inz;
	fread(&passo, sizeof(UINT), 1, src);
	fread(&tamanhoFiltro, sizeof(UINT), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createPool(cl, queue, passo, tamanhoFiltro, inx, iny, inz, entrada, params, error);
}