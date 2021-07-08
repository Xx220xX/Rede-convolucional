//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_TENSOR_H
#define CNN_GPU_TENSOR_H

#include <stdlib.h>
#include <math.h>
#include"../gpu/WrapperCL.h"
#include "../camadas/utils.h"


#ifdef LOG_CNN_TENSOR_MEMORY
#undef LOG_CNN_TENSOR_MEMORY
#define LOG_CNN_TENSOR_MEMORY(format, ...) printf("Tensor memory: ");printf(format,## __VA_ARGS__);printf("\n");
#else
#define LOG_CNN_TENSOR_MEMORY(format, ...)
#endif
typedef struct {
	int x, y, z;
} Ponto3d;

#ifdef TENSOR_TRANSPOSE
#define TensorMap(T, xx, yy, zz) (zz)*((T)->y*(T)->x)+(yy)*(T)->x+(xx)
#else
#define TensorMap(T, xx, yy, zz) (zz)*((T)->y*(T)->x)+(xx)*(T)->y+(yy)
#endif
typedef unsigned int UINT;
/***
 * Tensor armazena uma matriz 3D juntamente com os parametros dela
 */
typedef struct {
	cl_mem data;
	UINT bytes, x, y, z;
} *Tensor, typetensor, *TensorChar, typetensorchar;
typedef struct {
	int x, y, z;
	double *data;
} *TensorC, typeTensorC;

void printTensor(QUEUE q, Tensor t, FILE *f);

void fillTensor(Tensor t, cl_context context,QUEUE queue, size_t bytes, GPU_ERROR *error);

int TensorPutValues(QUEUE queue, Tensor t, void *data);

int TensorPutValuesOffSet(QUEUE queue, Tensor t, void *data, UINT ofset);

void TensorGetValues(QUEUE queue, Tensor t, void *data);

void TensorGetValuesOffset(QUEUE queue, Tensor t, int offset, void *data);

Tensor newTensor(cl_context context,QUEUE queue, UINT x, UINT y, UINT z, GPU_ERROR *error);

TensorChar newTensorChar(cl_context context,QUEUE queue, UINT x, UINT y, UINT z, GPU_ERROR *error);

Tensor newTensor4D(cl_context context,QUEUE queue, UINT x, UINT y, UINT z,
                   UINT l, GPU_ERROR *error);

void releaseTensor(Tensor *t);

void releaseTensorChar(TensorChar *t);

TensorC newTensorC(int x, int y, int z);

void dividirVetor(double *v, cl_mem m, size_t len, double value, Kernel funcNorm,
                  size_t max_works,
                  QUEUE queue);

void dividirVetorInt(unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, double value,
                     Kernel funcNorm,
                     size_t max_works,
                     QUEUE queue);

void int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len,
                      int nop,
                      Kernel func, QUEUE queue);

#endif //CNN_GPU_TENSOR_H
