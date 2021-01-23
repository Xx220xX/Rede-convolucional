//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_TENSOR_H
#define CNN_GPU_TENSOR_H

#include <stdlib.h>
#include <math.h>
#include"gpu/WrapperCL.h"

typedef struct {
    int x, y, z;
} Ponto3d;


#define TensorMap(T, xx, yy, zz) (zz)*((T)->y*(T)->x)+(yy)*(T)->x+(xx)
/***
 * Tensor armazena uma matriz 3D juntamente com os parametros dela
 */
typedef struct {
    cl_mem data;
    unsigned int bytes, x, y, z;
} *Tensor, typetensor, *TensorChar, typetensorchar;;


typedef struct {
    Ponto3d min, max;
} Range;

Tensor newTensor(cl_context context, unsigned int x, unsigned int y, unsigned int z, GPU_ERROR *error) {
    Tensor t = (Tensor) calloc(1, sizeof(typetensor));
    t->bytes = x * y * z * sizeof(double);

    t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, t->bytes, NULL, &error->error);
    if (!t->data) {
        error->error = -1;
        snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
    }
    if (!error->error) {
        snprintf(error->msg, 255, "nso foi possivel allocar memoria\n");
    }
    return t;
}

TensorChar newTensorChar(cl_context context, unsigned int x, unsigned int y, unsigned int z, GPU_ERROR *error) {
    TensorChar t = (Tensor) calloc(1, sizeof(typetensorchar));
    t->bytes = x * y * z * sizeof(char);

    t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, t->bytes, NULL, &error->error);
    if (!t->data) {
        error->error = -1;
        snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
    }
    if (!error->error) {
        snprintf(error->msg, 255, "nso foi possivel allocar memoria\n");
    }
    return t;
}

void releaseTensor(Tensor *t) {
    if (*t) {
        clReleaseMemObject((*t)->data);
        free((*t)->data);
        free(*t);
        *t = NULL;
    }
}

void releaseTensorChar(TensorChar *t) {
   releaseTensor(t);
}


Ponto3d mapeia_saida_entrada(int x, int y, int z, int z2, int passo) {
    Ponto3d saida = {x * passo, y * passo, z2};
    return saida;
}

int normaliza_range(double f, int max, int lim_min) {
    if (f < 0)return 0;
    if (f >= max - 1)return max - 1;
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



#endif //CNN_GPU_TENSOR_H
