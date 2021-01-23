//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADADROPOUT_H
#define CNN_GPU_CAMADADROPOUT_H

#include "string.h"
#include"Camada.h"
#include"Tensor.h"
#include <stdlib.h>
#include <float.h>

typedef unsigned int UINT;


typedef struct {
    Typecamada super;
    TensorChar hitmap;
    char flag_releaseInput;
    double p_ativacao;
} *CamadaDropOut, Typecamadadropout;

void releaseDropOut(CamadaDropOut *pc);

void corrigePesosDropOut(CamadaDropOut c);

void ativaDropOut(CamadaDropOut c);

void calc_gradsDropOut(CamadaDropOut c, Tensor GradNext);

Camada createDropOut(UINT inx, UINT iny, UINT inz, double p_ativacao, Tensor entrada) {
    CamadaDropOut c = (CamadaDropOut) calloc(1, sizeof(Typecamadadropout));
    c->super.gradsEntrada = newTensor(inx, iny, inz);
    if (!entrada) {
        c->super.entrada = newTensor(inx, iny, inz);
        c->flag_releaseInput = 1;
    } else {
        c->super.entrada = entrada;
    }
    c->super.saida = newTensor(inx, iny, inz);
    c->hitmap = newTensorChar(inx, iny, inz);
    c->p_ativacao = p_ativacao;

    c->super.release = (fv) releaseDropOut;
    c->super.ativa = (fv)ativaDropOut;
    c->super.calc_grads =(fvv) calc_gradsDropOut;
    c->super.corrige_pesos = (fv)corrigePesosDropOut;
    c->super.type = DROPOUT;
    return (Camada)c;
}

void releaseDropOut(CamadaDropOut *pc) {
    CamadaDropOut c = *pc;
    releaseTensor(&c->super.gradsEntrada);
    releaseTensor(&c->super.saida);
    releaseTensorChar(&c->hitmap);
    if (c->flag_releaseInput)releaseTensor(&c->super.entrada);
    *pc = 0;
}

void ativaDropOut(CamadaDropOut c) {
    char teste_ativa;
    for (int i = 0; i < c->super.entrada->tx * c->super.entrada->ty * c->super.entrada->tz; ++i) {
        teste_ativa = (char) ((rand() % RAND_MAX) / (double) RAND_MAX <= c->p_ativacao);
        c->hitmap->data[i] = teste_ativa;
        c->super.saida->data[i] = c->super.entrada->data[i] * teste_ativa;
    }
}


void corrigePesosDropOut(CamadaDropOut c) {}

void calc_gradsDropOut(CamadaDropOut c, Tensor GradNext) {
    for (int i = 0; i < c->super.entrada->tx * c->super.entrada->ty * c->super.entrada->tz; ++i) {
        c->super.gradsEntrada->data[i] = c->hitmap->data[i] * GradNext->data[i];
    }
}

#endif //CNN_GPU_CAMADADROPOUT_H
