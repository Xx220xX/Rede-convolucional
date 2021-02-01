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
Tensor newTensor4D(cl_context context, unsigned int x, unsigned int y, unsigned int z,unsigned int l, GPU_ERROR *error) {
    Tensor t = (Tensor) calloc(1, sizeof(typetensor));
    t->bytes = x * y * z * sizeof(double);

    t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, t->bytes*l, NULL, &error->error);
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



#endif //CNN_GPU_TENSOR_H
