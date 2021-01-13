//
// Created by gf154 on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_RELU_H
#define CNN_GPU_CAMADA_RELU_H

#include "Camada.h"
#include"Tensor.h"
#include <stdlib.h>

typedef struct {
    Typecamada super;
} *CamadaRelu, TypecamadaRelu;

void realeaseRelu(CamadaRelu *pc);

void ativaRelu(CamadaRelu c);

void corrige_pesosRelu(CamadaRelu );

void calc_gradsRelu(CamadaRelu c, Tensor GradNext);

Camada creatRelu(unsigned int inx, unsigned int iny, unsigned int inz,Tensor entrada) {
    CamadaRelu c = (CamadaRelu) calloc(1, sizeof(TypecamadaRelu));

    c->super.gradsEntrada = newTensor(inx, iny, inz);
    c->super.saida = newTensor(inx, iny, inz);
    c->super.entrada = entrada;
    if (!entrada) {
        c->super.entrada = newTensor(inx, iny, inz);
        c->super.flag_releaseInput = 1;
    }
    c->super.release = (fv) realeaseRelu;
    c->super.ativa = (fv) ativaRelu;
    c->super.calc_grads = (fvv) calc_gradsRelu;
    c->super.corrige_pesos = (fv) corrige_pesosRelu;
    c->super.type = RELU;
    return (Camada) c;
}

void realeaseRelu(CamadaRelu *pc) {
    CamadaRelu c = *pc;
    releaseTensor(&c->super.gradsEntrada);
    releaseTensor(&c->super.entrada);
    releaseTensor(&c->super.saida);
    free(c);
    *pc = NULL;
}

void ativaRelu(CamadaRelu c) {
    for (int i = 0; i < c->super.entrada->tx; i++)
        for (int j = 0; j < c->super.entrada->ty; j++)
            for (int z = 0; z < c->super.entrada->tz; z++) {
                double v = TensorAT(c->super.entrada, i, j, z);
                if (v < 0)
                    v = 0;
                TensorAT(c->super.saida, i, j, z) = v;
            }
}

void corrige_pesosRelu(CamadaRelu c) {}


void calc_gradsRelu(CamadaRelu c, Tensor GradNext) {
    for (int i = 0; i < c->super.entrada->tx; i++)
        for (int j = 0; j < c->super.entrada->ty; j++)
            for (int z = 0; z < c->super.entrada->tz; z++) {
                TensorAT(c->super.gradsEntrada, i, j, z) = (TensorAT(c->super.entrada, i, j, z) < 0) ? (0) :
                                                          (1 * TensorAT(GradNext, i, j, z));
            }
}


#endif //CNN_GPU_CAMADA_RELU_H
