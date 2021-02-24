//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_TENSOR_H
#define CNN_GPU_TENSOR_H

#include <stdlib.h>
#include <math.h>
#include"gpu/WrapperCL.h"

#ifdef LOG_CNN_TENSOR_MEMORY
#undef LOG_CNN_TENSOR_MEMORY
#define LOG_CNN_TENSOR_MEMORY(format, ...) printf("Tensor memory: ");printf(format,## __VA_ARGS__);printf("\n");
#else
#define LOG_CNN_TENSOR_MEMORY(format, ...)
#endif
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
} *Tensor, typetensor, *TensorChar, typetensorchar;

void fillTensor(Tensor t,cl_context context,size_t bytes,GPU_ERROR * error){
    t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &error->error);
    if (!t->data) {
        error->error = -1;
        snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
    }
    if (error->error) {
        snprintf(error->msg, 255, "nao foi possivel allocar memoria vram\n");

    }
    LOG_CNN_TENSOR_MEMORY("aloc (0x%X,0x%X)", t, t->data)
}

Tensor newTensor(cl_context context, unsigned int x, unsigned int y, unsigned int z, GPU_ERROR *error) {
    if (error->error)return NULL;
    if(x<=0|y<=0|z<=0) {
        error->error = -1;
    }
    Tensor t = (Tensor) calloc(1, sizeof(typetensor));
    t->bytes = x * y * z * sizeof(double);
    t->x = x;t->y = y;t->z = z;
    fillTensor(t,context,t->bytes,error);
    return t;
}

Tensor newTensor4D(cl_context context, unsigned int x, unsigned int y, unsigned int z, unsigned int l, GPU_ERROR *error) {
    if (error->error)return NULL;
    Tensor t = (Tensor) calloc(1, sizeof(typetensor));
    t->bytes = x * y * z * sizeof(double);
    t->x = x;t->y = y;t->z = z;
    fillTensor(t,context, t->bytes * l, error);
    return t;
}

TensorChar newTensorChar(cl_context context, unsigned int x, unsigned int y, unsigned int z, GPU_ERROR *error) {
    if (error->error)return NULL;
    TensorChar t = (Tensor) calloc(1, sizeof(typetensorchar));
    t->bytes = x * y * z * sizeof(char);
    t->x = x;t->y = y;t->z = z;
    fillTensor(t,context, t->bytes,error);
    return t;
}

void releaseTensor(Tensor *t) {
    if (*t) {
        LOG_CNN_TENSOR_MEMORY("free (0x%X,0x%X)", *t, (*t)->data)
        if ((*t)->data)
            clReleaseMemObject((*t)->data);
        free(*t);
        *t = NULL;
    }
}

void releaseTensorChar(TensorChar *t) {
    releaseTensor(t);
}


#endif //CNN_GPU_TENSOR_H
