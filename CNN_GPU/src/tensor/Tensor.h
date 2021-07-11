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
/**
 * Armazena as três dimensões de um ponto
 * @Atributtes x dimensão x do Ponto
 * @Atributtes y dimensão y do Ponto
 * @Atributtes z dimensão z do Ponto
 */
typedef struct Ponto3d {
	unsigned int x, y, z;
} Ponto3d;
/**
 * Mapeia um ponto 3d no vetor
 */
#define TensorMap(T, xx, yy, zz) (zz)*((T)->y*(T)->x)+(xx)*(T)->y+(yy)
#define TENSOR_NCPY 0x00
#define TENSOR_HOST 0x01
#define TENSOR_HSTA 0x02
#define TENSOR_SVM  0x03


typedef unsigned int UINT;
/***
 * Tensor armazena uma matriz 4D juntamente com os parametros dela
 * @Atributtes data memoria no driver
 * @Atributtes host memoria na ram
 * @Atributtes flags Tipo de memoria
 * @Atributtes bytes total armazenado
 * @Atributtes x dimensão x do tensor
 * @Atributtes y dimensão y do tensor
 * @Atributtes z dimensão z do tensor
 * @Atributtes w dimensão w do tensor
 * @flags  TENSOR_NCPY Não faz nenhuma copia, toda memoria é armazenada no driver
 * @flags TENSOR_HOST Faz a copia com um ponteiro host que pode ser usando enquanto o kernel está executando
 * @flags TENSOR_HSTA Faz a copia mas o driver faz alocação dos recursos
 * @flags TENSOR_SVM Utiliza a memoria compartilhada host=data
 */
typedef struct typetensor {
	cl_mem data;
	void *host;
	char flag;
	UINT bytes, x, y, z, w;
} *Tensor, typetensor, *TensorChar;


typedef struct {
	int x, y, z, l;
	double *data;
} *TensorC, typeTensorC;


Tensor newTensor(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char usehost, Exception *error);

TensorChar newTensorChar(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char usehost, Exception *error);

Tensor newTensor4D(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, UINT l, char usehost, Exception *error);

void printTensor(QUEUE q, Tensor t, FILE *f);

int TensorFill(QUEUE queue, Tensor t, char patern);

int TensorFillOffSet(QUEUE queue, Tensor t,  char patern, UINT offset);

int TensorPutValues(QUEUE queue, Tensor t, void *data);

int TensorPutValuesOffSet(QUEUE queue, Tensor t, void *data, UINT ofset);

int TensorGetValues(QUEUE queue, Tensor t, void *data);

int TensorGetValuesOffset(QUEUE queue, Tensor t, void *data, unsigned int offset);


void releaseTensor(Tensor *t);

void releaseTensorChar(TensorChar *t);

TensorC newTensorC(int x, int y, int z);

int dividirVetor(double *v, cl_mem m, size_t len, double value, Kernel funcNorm,
                 size_t max_works,
                 QUEUE queue);

int dividirVetorInt(unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, double value,
                    Kernel funcNorm,
                    size_t max_works,
                    QUEUE queue);

int int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len,
                     int nop,
                     Kernel func, QUEUE queue);

#endif //CNN_GPU_TENSOR_H
