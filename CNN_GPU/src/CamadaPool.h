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

Camada createPool(UINT passo, UINT tamanhoFiltro, UINT inx, UINT iny, UINT inz, Tensor entrada, Params *params){
    CamadaPool c = (CamadaPool) calloc(1, sizeof(Typecamadapool));
    c->passo = passo;
    c->tamanhoFiltro = tamanhoFiltro;
    if (!entrada) {
        c->super.entrada = newTensor(inx, iny, inz);
        c->super.flag_releaseInput = 1;
    } else {
        c->super.entrada = entrada;
    }
    c->super.gradsEntrada = newTensor(inx, iny, inz);
    c->super.saida = newTensor((inx - tamanhoFiltro) / passo + 1, (iny - tamanhoFiltro) / passo + 1, inz);
    c->super.release = (fv) releasePool;
    c->super.ativa = (fv) ativaPool;
    c->super.corrige_pesos = (fv) corrige_pesosPool;
    c->super.calc_grads = (fvv) calc_gradsPool;
    c->super.parametros = params;
    c->super.type = POOL;

    c->kernelPoolAtiva = new_Kernel(cl->program, "poolativa", 8, VOID_P, VOID_P, INT, INT, INT, INT, INT, INT);
    c->kernelPoolCalcGrads = new_Kernel(cl->program, "poolCalcGrads", 11, VOID_P, VOID_P, VOID_P, VOID_P,
                                        INT, INT, INT, INT, INT, INT, INT);
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
    int error = 0, id = 0;
    size_t global, local, resto;
    Params *parametros = c->super.parametros;
    call_kernel(c->super.saida->x*c->super.saida->y*c->super.saida->z,
                Kernel_putArgs(&c->kernelPoolAtiva, 8, &c->super.entrada->data, &c->super.saida->data, c->tamanhoFiltro,&c->passo,
                               &c->super.saida->x, &c->super.saida->y, &c->super.saida->z, &id);
    error = clEnqueueNDRangeKernel(c->super.queue, c->kernelReluCalcGrads.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    PERRW(error, "falha ao chamar kernel ativa dropout")
    );
}

void corrige_pesosPool(CamadaPool c) {}


void calc_gradsPool(CamadaPool c, Tensor GradNext) {
    Range range={0};
    double somaErro = 0;
    int minx, miny;
    double testeMax;
    call_kernel(c->super.saida->x*c->super.saida->y*c->super.saida->z,
                Kernel_putArgs(&c->kernelPoolCalcGrads, 11, &c->super.entrada->data, &c->super.gradsEntrada, Gradnext->data,
                &c->super.saida->data, c->tamanhoFiltro, &c->passo, &c->super.entrada->x, &c->super.entrada->y,
                &c->super.saida->x, &c->super.saida->y, &id);
    error = clEnqueueNDRangeKernel(c->super.queue, c->kernelReluCalcGrads.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    PERRW(error, "falha ao chamar kernel ativa dropout")
    );

}

#endif //CNN_GPU_CAMADAPOOL_H
