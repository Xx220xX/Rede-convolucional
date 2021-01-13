//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADAFULLCONNECT_H
#define CNN_GPU_CAMADAFULLCONNECT_H

#include "string.h"
#include"Camada.h"
#include"Tensor.h"
#include <stdlib.h>
#include <float.h>

#include"funcoesDeAtivacao.h"

typedef unsigned int UINT;

typedef struct {
    Typecamada super;
    Tensor input;
    Tensor pesos;
    Tensor grad;
    Tensor oldgrad;
    // funcao de ativacao e sua derivada
    dfd fa, dfa;
} *CamadaFullConnect, Typecamadafullconnect;
void releaseFullConnect(CamadaFullConnect *pc);


void corrigePesosFullConnect(CamadaFullConnect c);

void ativaFullConnect(CamadaFullConnect c);

void calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext);

Camada createFullConnect(UINT inx, UINT iny, UINT inz, UINT tamanhoSaida, Tensor entrada,int funcaoDeAtivacao) {
    CamadaFullConnect c = (CamadaFullConnect) calloc(1, sizeof(Typecamadafullconnect));
    c->super.gradsEntrada = newTensor(inx, iny, inz);
    if (!entrada) {
        c->super.entrada = newTensor(inx, iny, inz);
        c->super.flag_releaseInput = 1;
    } else {
        c->super.entrada = entrada;
    }

    c->super.saida = newTensor(tamanhoSaida, 1, 1);
    c->input = newTensor(tamanhoSaida, 1, 1);
    c->grad = newTensor(tamanhoSaida, 1, 1);
    c->oldgrad = newTensor(tamanhoSaida, 1, 1);

    c->pesos = newTensor(inx * iny * inz, tamanhoSaida, 1);
    int valmax = inx * iny * inz;

    for (int i = 0; i < tamanhoSaida; ++i) {
        for (int j = 0; j < valmax; ++j) {
            TensorAT(c->pesos, j, i, 0) = 2.19722 / (valmax) * rand() / (double) RAND_MAX;
        }
    }

    c->super.release = (fv) releaseFullConnect;
    c->super.ativa = (fv) ativaFullConnect;
    c->super.calc_grads = (fvv) calc_gradsFullConnect;
    c->super.corrige_pesos = (fv) corrigePesosFullConnect;
    c->super.type = FULLCONNECT;
    c->fa = funcoesDeAtivacao[funcaoDeAtivacao];
    c->fa = funcoesDeAtivacao[funcaoDeAtivacao+FLAGDIF];

    return (Camada) c;
}


void releaseFullConnect(CamadaFullConnect *pc) {
    CamadaFullConnect c = *pc;
    if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
    releaseTensor(&c->super.gradsEntrada);
    releaseTensor(&c->pesos);
    releaseTensor(&c->grad);
    releaseTensor(&c->oldgrad);
    releaseTensor(&c->super.saida);
    releaseTensor(&c->input);
    free(c);
    *pc = 0;

}

void ativaFullConnect(CamadaFullConnect c) {
    double valorEntrada;
    int m;
    for (int n = 0; n < c->super.saida->tx; ++n) {
        valorEntrada = 0;
        for (int i = 0; i < c->super.entrada->tx; ++i) {
            for (int j = 0; j < c->super.entrada->ty; ++j) {
                for (int z = 0; z < c->super.entrada->tx; ++z) {
                    m = i * (c->super.entrada->ty * c->super.entrada->tz) + j * c->super.entrada->tz + z;
                    valorEntrada += TensorAT(c->super.entrada, i, j, z) * TensorAT(c->pesos, m, n, 0);
                }
            }
        }
        c->input->data[n] = valorEntrada;
        c->super.saida->data[n] = c->fa(valorEntrada);
    }
}

void corrigePesosFullConnect(CamadaFullConnect c) {
    int m;
    double w;
    double tmp;
    for (int n = 0; n < c->super.saida->tx; ++n) {
        for (int i = 0; i < c->super.entrada->tx; ++i) {
            for (int j = 0; j < c->super.entrada->ty; ++j) {
                for (int z = 0; z < c->super.entrada->tx; ++z) {
                    m = i * (c->super.entrada->ty * c->super.entrada->tz) + j * c->super.entrada->tz + z;
                    w = TensorAT(c->pesos, m, n, 0);
                    tmp = c->grad->data[n] + c->oldgrad->data[n] * c->super.parametros->momento;
                    w -= c->super.parametros->hitLearn *
                         (tmp * TensorAT(c->super.entrada, i, j, z) + w * c->super.parametros->decaimentoDePeso);
                    TensorAT(c->pesos, m, n, 0) = w;
                }
            }
        }
        c->oldgrad->data[n] = c->grad->data[n] + c->oldgrad->data[n] * c->super.parametros->momento;
    }

}

void calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext) {
    memset(c->super.gradsEntrada, 0, c->super.gradsEntrada->tx * c->super.gradsEntrada->ty * c->super.gradsEntrada->tz);
    int m;
    for (int n = 0; n < c->super.saida->tx; ++n) {
        c->grad->data[n] = TensorAT(GradNext, n, 0, 0) * c->dfa(c->input->data[n]);
        for (int n = 0; n < c->super.saida->tx; ++n) {
            for (int i = 0; i < c->super.entrada->tx; ++i) {
                for (int j = 0; j < c->super.entrada->ty; ++j) {
                    for (int z = 0; z < c->super.entrada->tx; ++z) {
                        m = i * (c->super.entrada->ty * c->super.entrada->tz) + j * c->super.entrada->tz + z;
                        TensorAT(c->super.gradsEntrada, i, j, z) += c->grad->data[n] * TensorAT(c->pesos, m, n, 0);
                    }
                }
            }
        }
    }
}

#endif //CNN_GPU_CAMADAFULLCONNECT_H
