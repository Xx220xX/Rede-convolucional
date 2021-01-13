//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONV_H
#define CNN_GPU_CAMADACONV_H

#include "string.h"
#include"Camada.h"
#include"Tensor.h"
#include <stdlib.h>

typedef unsigned int UINT;


typedef struct {
    Typecamada super;
    Tensor *filtros;
    Tensor *grad_filtros;
    Tensor *grad_filtros_old;
    UINT passo, tamanhoFiltro, numeroFiltros;

} *CamadaConv, Typecamadaconv;
void calc_gradsConv(CamadaConv c, Tensor Gradnext);
void releaseConv(CamadaConv *pc);

void ativaConv(CamadaConv c);

void corrige_pesosConv(CamadaConv c);

Camada createConv(UINT passo, UINT tamanhoFiltro, UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params *params) {
    CamadaConv c = (CamadaConv) calloc(1, sizeof(Typecamadaconv));
    c->passo = passo;
    c->tamanhoFiltro = tamanhoFiltro;
    c->numeroFiltros = numeroFiltros;
    c->super.gradsEntrada = newTensor(inx, iny, inz);
    c->super.entrada = entrada;
    if (!entrada) {
        c->super.entrada = newTensor(inx, iny, inz);
        c->super.flag_releaseInput = 1;
    }
    c->super.saida = newTensor((inx - tamanhoFiltro) / passo + 1, (iny - tamanhoFiltro) / passo + 1,
                               numeroFiltros);
    c->filtros = (Tensor *) calloc(numeroFiltros, sizeof(Tensor));
    c->grad_filtros = (Tensor *) calloc(numeroFiltros, sizeof(Tensor));
    c->grad_filtros_old = (Tensor *) calloc(numeroFiltros, sizeof(Tensor));
    Tensor tmp;
    UINT maxVal = tamanhoFiltro * tamanhoFiltro * inz;
    for (int a = 0; a < numeroFiltros; a++) {
        tmp = newTensor(tamanhoFiltro, tamanhoFiltro, inz);

        for (int i = 0; i < tamanhoFiltro; ++i) {
            for (int j = 0; j < tamanhoFiltro; ++j) {
                for (int z = 0; z < inz; ++z) {
                    TensorAT(tmp, i, j, z) = 1.0 / maxVal * (rand() / ((double) RAND_MAX));
                }
            }
        }
        c->filtros[a] = tmp;
        c->grad_filtros[a] = newTensor(tamanhoFiltro, tamanhoFiltro, inz);
    }
    c->super.release = (fv) releaseConv;
    c->super.ativa = (fv) ativaConv;
    c->super.calc_grads = (fvv) calc_gradsConv;
    c->super.corrige_pesos = (fv) corrige_pesosConv;
    c->super.parametros = params;
    c->super.type = CONV;
    return (Camada) c;
}

void releaseConv(CamadaConv *pc) {
    CamadaConv c = *pc;
    releaseTensor(&c->super.gradsEntrada);
    if (c->super.flag_releaseInput)
        releaseTensor(&c->super.entrada);
    releaseTensor(&c->super.saida);
    for (int a = 0; a < c->numeroFiltros; a++) {
        releaseTensor(c->filtros + a);
        releaseTensor(c->grad_filtros + a);
        releaseTensor(c->grad_filtros_old + a);
    }
    free(c->filtros);
    free(c);
    *pc = NULL;
}

void ativaConv(CamadaConv c) {
    Tensor filtro;
    Ponto3d mapeado;
    double sum, f, v;
    Tensor entrada = c->super.entrada;
    //iteraçao nos filtros
    for (int filtrok = 0; filtrok < c->numeroFiltros; filtrok++) {
        //seleciona o filtro
        filtro = c->filtros[filtrok];

        for (int x = 0; x < c->super.saida->tx; x++) {
            for (int y = 0; y < c->super.saida->ty; y++) {
                //converte as cordenadas da saida ´para a entrada
                //P*passo
                mapeado = mapeia_saida_entrada(x, y, 0, 0, c->passo);
                sum = 0;
                // soma convolutiva
                for (int i = 0; i < c->tamanhoFiltro; i++) {
                    for (int j = 0; j < c->tamanhoFiltro; j++) {
                        for (int z = 0; z < entrada->tz; z++) {
                            f = TensorAT(filtro, i, j, z);
                            v = TensorAT(entrada, mapeado.x + i, mapeado.y + j, z);
                            sum += f * v;
                        }
                    }
                }
                TensorAT(c->super.saida, x, y, filtrok) = sum;
            }

        }
    }
}


void corrige_pesosConv(CamadaConv c) {
    double w = 0, m = 0;
    double grad, oldGrad;
    Params *parametros = c->super.parametros;
    for (int a = 0; a < c->numeroFiltros; a++) {
        for (int i = 0; i < c->tamanhoFiltro; i++) {
            for (int j = 0; j < c->tamanhoFiltro; j++) {
                for (int z = 0; z < c->super.entrada->tz; z++) {
                    w = TensorAT(c->filtros[a], i, j, z);
                    grad = TensorAT(c->grad_filtros[a], i, j, z);
                    oldGrad = TensorAT(c->grad_filtros_old[a], i, j, z);
                    m = grad + oldGrad * parametros->momento;
                    TensorAT(c->filtros[a], i, j, z) =
                            w - parametros->hitLearn * (m * TensorAT(c->super.entrada, i, j, z) +
                                                        w * parametros->decaimentoDePeso);
                    TensorAT(c->grad_filtros_old[a], i, j, z) =
                            grad + oldGrad * parametros->momento;
                }
            }
        }
    }
}


void calc_gradsConv(CamadaConv c, Tensor Gradnext) {
    // zerar o gradiente
    for (int k = 0; k < c->numeroFiltros; k++) {
        memset((void *) c->grad_filtros[k]->data, 0, c->tamanhoFiltro * c->tamanhoFiltro * c->grad_filtros[k]->tz);
    }
    Range range = {0};
    double somaErro = 0;
    int minX, minY;
    double pesoAplicado;
    for (int x = 0; x < c->super.entrada->tx; x++) {
        for (int y = 0; y < c->super.entrada->ty; y++) {
            range = mapeia_entrada_saida(x, y, c->passo, c->tamanhoFiltro, c->super.saida, c->numeroFiltros);
            for (int z = 0; z < c->super.entrada->tz; z++) {
                somaErro = 0;
                for (int i = range.min.x; i <= range.max.x; i++) {
                    minX = i * c->passo;
                    for (int j = range.min.y; j <= range.max.y; j++) {
                        minY = j * c->passo;
                        for (int k = range.min.z; k <= range.max.z; k++) {
                            pesoAplicado = TensorAT(c->filtros[k], x - minX, y - minY, z);
                            somaErro += pesoAplicado * TensorAT(Gradnext, i, j, k);
                            TensorAT(c->grad_filtros[k], x - minX, y - minY, z) +=
                                    TensorAT(c->super.entrada, x, y, z) * TensorAT(Gradnext, i, j, k);
                        }
                    }
                }
                TensorAT(c->super.gradsEntrada, x, y, z) = somaErro;
            }
        }
    }
}

#endif //CNN_GPU_CAMADACONV_H
