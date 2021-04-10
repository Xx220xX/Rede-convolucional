//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_TENSOR_H
#define CNN_GPU_TENSOR_H

#include <stdlib.h>
#include <math.h>

typedef struct {
    int x, y, z;
} Ponto3d;


#define TensorAT(T, x, y, z) (T)->data[(z)*((T)->ty*(T)->tx)+(y)*(T)->tx+(x)]
typedef struct {
    double *data;
    unsigned int tx, ty, tz;

} *Tensor, typetensor;

typedef struct {
    char *data;
    unsigned int tx, ty, tz;

} *TensorChar, typetensorchar;


typedef struct {
    Ponto3d min, max;
} Range;

Tensor newTensor(unsigned int x, unsigned int y, unsigned int z) {
    Tensor t = (Tensor) calloc(1, sizeof(typetensor));
    t->tx = x;
    t->ty = y;
    t->tz = z;
    t->data = (double *) calloc(x * y * z, sizeof(double));
    return t;
}

TensorChar newTensorChar(unsigned int x, unsigned int y, unsigned int z) {
    TensorChar t = (TensorChar) calloc(1, sizeof(typetensorchar));
    t->tx = x;
    t->ty = y;
    t->tz = z;
    t->data = (char *) calloc(x * y * z, sizeof(char));
    return t;
}

void releaseTensor(Tensor *t) {
    if (*t) {
        free((*t)->data);
        free(*t);
        *t = NULL;
    }
}

void releaseTensorChar(TensorChar *t) {
    if (*t) {
        free((*t)->data);
        free(*t);
        *t = NULL;
    }
}


Ponto3d mapeia_saida_entrada(int x, int y, int z, int z2, int passo) {
    Ponto3d saida = {x * passo, y * passo, z2};
    return saida;
}

int normaliza_range(double f, int max, int lim_min) {
    if (f < 0)return 0;
    if (f >= max - 1)return max-1;
    if (lim_min) return ceil(f);
    else return floor(f);
}

Range mapeia_entrada_saida(int x, int y, int passo, int tamanhoFiltro, Tensor saida, int numeroFiltros) {
    double a = x, b = y;
    Range r = {0};
    r.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saida->tx, 1);
    r.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saida->ty, 1);
    r.max.x = normaliza_range(a / passo, saida->tx, 0);
    r.max.y = normaliza_range(b / passo, saida->ty, 0);
    r.max.z = numeroFiltros - 1;
    return r;
}


void printTensor(Tensor t) {
    for (int z = 0; z < t->tz; ++z) {
        printf("[Dim%d]\n", z);
        for (int x = 0; x < t->tx; ++x) {
            for (int y = 0; y < t->ty; ++y) {
                printf("%.2f \t",TensorAT(t,x,y,z));
            }
            printf("\n");
        }
    }
    printf("\n");
}

void printPonto3D(Ponto3d p){
    printf("(%.4lf,%.4lf,%.4lf)\n",p.x,p.y,p.z);
}
#endif //CNN_GPU_TENSOR_H
