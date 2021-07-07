//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaPoolAv.h"
const char *tostringPoolAv(CamadaPoolAv c){
    if(c->super.__string__ != NULL)free(c->super.__string__);
    c->super.__string__ = (char *) calloc(1000, sizeof(char));
    int len = snprintf(c->super.__string__, 1000,
                       "Average Pooling  Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
                       "\tStep %u\n",

                       c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
                       c->super.saida->x, c->super.saida->y, c->super.saida->z,
                       c->passo
    );
    len+=1;
    c->super.__string__ = realloc(c->super.__string__, sizeof (char) * len);
    return c->super.__string__;
}
Camada createPoolAv(WrapperCL *cl, cl_command_queue queue, UINT passo, UINT tamanhoFiltro, UINT inx, UINT iny, UINT inz,
           Tensor entrada, Params params,
           GPU_ERROR *error) {
	CamadaPoolAv c = (CamadaPoolAv) calloc(1, sizeof(TypecamadaPoolAv));
	c->passo = passo;
	c->tamanhoFiltro = tamanhoFiltro;
	__newCamada__((Camada) c, cl, POOLAV, entrada, queue, params, inx, iny, inz, (inx - tamanhoFiltro) / passo + 1,
	              (iny - tamanhoFiltro) / passo + 1, inz,
	              error);
    c->super.toString = (fch) tostringPoolAv;
	c->super.release = (fv) releasePoolAv;
	c->super.ativa = (fv) ativaPoolAv;
	c->super.corrige_pesos = (fv) corrige_pesosPoolAv;
	c->super.calc_grads = (fvv) calc_gradsPoolAv;
	c->super.parametros = params;
	c->super.salvar = (fsl) salvarPoolAv;

	c->kernelPoolAvAtiva = new_Kernel(cl->program,error, "PoolAvativa", 9, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT, K_INT, K_INT,
	                                K_INT, K_INT);
	c->kernelPoolAvCalcGrads = new_Kernel(cl->program,error, "PoolAvCalcGrads", 12, K_VOID_P, K_VOID_P, K_VOID_P, K_VOID_P,
	                                    K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT, K_INT);
	return (Camada) c;
}

void releasePoolAv(CamadaPoolAv *pc) {
	CamadaPoolAv c = *pc;
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
    if(c->super.__string__ != NULL){
        free(c->super.__string__);
    }
	releaseTensor(&c->super.gradsEntrada);
	releaseTensor(&c->super.saida);
	releaseKernel(&c->kernelPoolAvCalcGrads);
	releaseKernel(&c->kernelPoolAvAtiva);
	free(c);
	*pc = NULL;
}

void ativaPoolAv(CamadaPoolAv c) {
	kernel_run_recursive(&c->kernelPoolAvAtiva, c->super.queue, c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     *c->super.max_works,
	                     &c->super.entrada->data, &c->super.saida->data, &c->tamanhoFiltro, &c->passo,
	                     &c->super.saida->x, &c->super.saida->y, &c->super.entrada->x, &c->super.entrada->y);
}

void corrige_pesosPoolAv(CamadaPoolAv c) {}


void calc_gradsPoolAv(CamadaPoolAv c, Tensor GradNext) {
	kernel_run_recursive(&c->kernelPoolAvCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	                     *c->super.max_works,
	                     &c->super.entrada->data, &c->super.gradsEntrada->data,
	                     &GradNext->data,
	                     &c->super.saida->data, &c->tamanhoFiltro, &c->passo, &c->super.entrada->x,
	                     &c->super.entrada->y, &c->super.entrada->z,
	                     &c->super.saida->x, &c->super.saida->y);
}

void salvarPoolAv(WrapperCL *cl, CamadaPoolAv c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->passo, sizeof(UINT), 1, dst);
	fwrite(&c->tamanhoFiltro, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
}

Camada carregarPoolAv(WrapperCL *cl, FILE *src, cl_command_queue queue, Tensor entrada,
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
	return createPoolAv(cl,queue, passo, tamanhoFiltro, inx, iny, inz, entrada, params, error);
}
