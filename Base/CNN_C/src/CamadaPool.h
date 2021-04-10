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

} *CamadaPool, Typecamadapool;

void releasePool(CamadaPool *pc);

void ativaPool(CamadaPool c);

void corrige_pesosPool(CamadaPool c);

void calc_gradsPool(CamadaPool c, Tensor GradNext);

Camada createPool(UINT passo, UINT tamanhoFiltro, UINT inx, UINT iny, UINT inz, Tensor entrada, Params *params) {
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

    return (Camada) c;
}

void releasePool(CamadaPool *pc) {
    CamadaPool c = *pc;
    if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
    releaseTensor(&c->super.gradsEntrada);
    releaseTensor(&c->super.saida);
    *pc = NULL;
}

void ativaPool(CamadaPool c) {
    Ponto3d mapeado;
    double mval;
    double v;
    Params *parametros = c->super.parametros;
    for (int x = 0; x < c->super.saida->tx; ++x) {
        for (int y = 0; y < c->super.saida->ty; ++y) {
            for (int z = 0; z < c->super.saida->tz; ++z) {
                mapeado = mapeia_saida_entrada(x, y, 0, 0, c->passo);
                mval = -DBL_MAX;
                for (int i = 0; i < c->tamanhoFiltro; ++i) {
                    for (int j = 0; j < c->tamanhoFiltro; ++j) {
                        v = TensorAT(c->super.entrada, mapeado.x + i, mapeado.y + j, z);
                        if (v > mval)mval = v;
                    }
                }
                TensorAT(c->super.saida, x, y, z) = mval;
            }
        }
    }
}

void corrige_pesosPool(CamadaPool c) {}

Range mapeia_entrada_saidaPool(int x, int y, int tamanhoFiltro, int passo, Tensor saida) {
    float a = x, b = y;
    Range g = {0};
    g.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saida->tx, 1);
    g.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saida->ty, 1);
    g.max.x = normaliza_range(a / passo, saida->tx, 0);
    g.max.y = normaliza_range(b / passo, saida->ty, 0);
    g.max.z = saida->tz - 1;
    return g;

}

void calc_gradsPool(CamadaPool c, Tensor GradNext) {
    Range range={0};
    double somaErro = 0;
    int minx, miny;
    double testeMax;
    FOR2D(x,y,c->super.entrada->tx,c->super.entrada->ty) {
            range = mapeia_entrada_saidaPool(x, y, c->tamanhoFiltro, c->passo, c->super.saida);
            for (int z = 0; z < c->super.entrada->tz; ++z) {
                somaErro = 0;
                for (int i = range.min.x; i <= range.max.x; ++i) {
                    minx = i * c->passo;
                    for (int j = range.min.y; j <= range.max.y; ++j) {
                        miny = j * c->passo;
                        testeMax = TensorAT(c->super.entrada, x, y, z) == TensorAT(c->super.saida, i, j, z);
                        somaErro += testeMax * TensorAT(GradNext, i, j, z);
                    }
                }
                TensorAT(c->super.gradsEntrada, x, y, z) = somaErro;
            }
        }

}

#endif //CNN_GPU_CAMADAPOOL_H
