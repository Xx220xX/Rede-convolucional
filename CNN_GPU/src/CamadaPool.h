//
// Created by Xx220xX on 25/10/2020.
//

#ifndef CNN_GPU_CAMADAPOOL_H
#define CNN_GPU_CAMADAPOOL_H

#include "string.h"
#include"Camada.h"
#include"Tensor.h"
#include <stdlib.h>
#include <float.h>

typedef unsigned int UINT;


typedef struct {
	Typecamada super;
	UINT passo;
	UINT tamanhoFiltro;

	Kernel kernelPoolAtiva;
	Kernel kernelPoolCalcGrads;
} *CamadaPool, Typecamadapool;

void releasePool(CamadaPool *pc);

void ativaPool(CamadaPool c);

void corrige_pesosPool(CamadaPool c);

void calc_gradsPool(CamadaPool c, Tensor GradNext);

void salvarPool(WrapperCL *cl, CamadaPool c, FILE *dst, GPU_ERROR *error);

Camada
createPool(WrapperCL *cl, UINT passo, UINT tamanhoFiltro, UINT inx, UINT iny, UINT inz, Tensor entrada, Params *params,
           GPU_ERROR *error) {
	CamadaPool c = (CamadaPool) calloc(1, sizeof(Typecamadapool));
	c->passo = passo;
	c->tamanhoFiltro = tamanhoFiltro;
	if (!entrada) {
		c->super.entrada = newTensor(cl->context, inx, iny, inz, error);
		c->super.flag_releaseInput = 1;
	} else {
		c->super.entrada = entrada;
	}
	c->super.gradsEntrada = newTensor(cl->context, inx, iny, inz, error);
	c->super.saida = newTensor(cl->context, (inx - tamanhoFiltro) / passo + 1, (iny - tamanhoFiltro) / passo + 1, inz,
	                           error);
	c->super.release = (fv) releasePool;
	c->super.ativa = (fv) ativaPool;
	c->super.corrige_pesos = (fv) corrige_pesosPool;
	c->super.calc_grads = (fvv) calc_gradsPool;
	c->super.parametros = params;
	c->super.type = POOL;
	c->super.salvar = (fsl) salvarPool;

	c->kernelPoolAtiva = new_Kernel(cl->program, "poolativa", 9, VOID_P, VOID_P, INT, INT, INT, INT, INT, INT, INT);
	c->kernelPoolCalcGrads = new_Kernel(cl->program, "poolCalcGrads", 12, VOID_P, VOID_P, VOID_P, VOID_P,
	                                    INT, INT, INT, INT, INT, INT, INT, INT);
	return (Camada) c;
}

void releasePool(CamadaPool *pc) {
	CamadaPool c = *pc;
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
	releaseTensor(&c->super.gradsEntrada);
	releaseTensor(&c->super.saida);
	Kernel_release(&c->kernelPoolCalcGrads);
	Kernel_release(&c->kernelPoolAtiva);
	free(c);
	*pc = NULL;
}

void ativaPool(CamadaPool c) {
	LOG_CNN_KERNELCALL("ativa pool: ativaPool");
	kernel_run_recursive(&c->kernelPoolAtiva, c->super.queue, c->super.saida->x * c->super.saida->y * c->super.saida->z,
	                     max_works,
	                     &c->super.entrada->data, &c->super.saida->data, &c->tamanhoFiltro, &c->passo,
	                     &c->super.saida->x, &c->super.saida->y, &c->super.entrada->x, &c->super.entrada->y);
}

void corrige_pesosPool(CamadaPool c) {}


void calc_gradsPool(CamadaPool c, Tensor GradNext) {
	LOG_CNN_KERNELCALL("calcgrads pool: calcgrad")
	kernel_run_recursive(&c->kernelPoolCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z, max_works,
	                     &c->super.entrada->data, &c->super.gradsEntrada->data,
	                     &GradNext->data,
	                     &c->super.saida->data, &c->tamanhoFiltro, &c->passo, &c->super.entrada->x,
	                     &c->super.entrada->y, &c->super.entrada->z,
	                     &c->super.saida->x, &c->super.saida->y);
}

void salvarPool(WrapperCL *cl, CamadaPool c, FILE *dst, GPU_ERROR *error) {
	LOG_CNN_SALVE_LAYERS("salvando pool")
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->passo, sizeof(UINT), 1, dst);
	fwrite(&c->tamanhoFiltro, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	LOG_CNN_SALVE_LAYERS("salvou com erro %d: %s", error->error, error->msg)

}

Camada carregarPool(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error) {
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
	return createPool(cl, passo, tamanhoFiltro, inx, iny, inz, entrada, params, error);

}

#endif //CNN_GPU_CAMADAPOOL_H
