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
 * Mapeia um ponto 3d no vetor
 */
#define Tensor_Map(T, xx, yy, zz) (zz)*((T)->y*(T)->x)+(xx)*(T)->y+(yy)

/// Não faz nenhuma copia, toda memoria é armazenada no driver.
#define TENSOR_NCPY 0x00
///Faz a copia com um ponteiro host que pode ser usando enquanto o kernel está executando.
#define TENSOR_HOST 0x01
///Faz a copia com um ponteiro host que pode ser usando enquanto o kernel está executando, o driver faz alocação do host.
#define TENSOR_SMEM 0x02


typedef unsigned int UINT;
/**
 * Armazena as três dimensões de um ponto
 * @Atributtes x dimensão x do Ponto
 * @Atributtes y dimensão y do Ponto
 * @Atributtes z dimensão z do Ponto
 */
typedef struct Ponto3d {
	UINT x, y, z;
} Ponto;
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
 * @flags TENSOR_NCPY Não faz nenhuma copia, toda memoria é armazenada no driver
 * @flags TENSOR_HOST Faz a copia com um ponteiro host que pode ser usando enquanto o kernel está executando
 * @flags TENSOR_HSTA Faz a copia mas o driver faz alocação dos recursos
 */
typedef struct typetensor {
	cl_mem data;
	UINT bytes;
	UINT x;
	UINT y;
	UINT z;
	UINT w;
	void *host;
	char flag;
	cl_context context;

} *Tensor, typetensor, *TensorChar;


Tensor newTensor(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char tensor_flag, Exception *error);

TensorChar newTensorChar(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char tensor_flag, Exception *error);

Tensor newTensor4D(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, UINT l, char tensor_flag, Exception *error);

void printTensor(QUEUE q, Tensor t, FILE *f);

int TensorFill(QUEUE queue, Tensor t, char patern);

int TensorFillOffSet(QUEUE queue, Tensor t, char patern, size_t offset);

int TensorPutValues(QUEUE queue, Tensor t, void *data);

int TensorPutValuesOffSet(QUEUE queue, Tensor t, void *data, size_t ofset);

int TensorGetValues(QUEUE queue, Tensor t, void *data);

int TensorGetValuesOffSet(QUEUE queue, Tensor t, void *data, size_t offset);

int TensorGetValuesMem(QUEUE queue, Tensor t, void *data, size_t bytes);

int TensorGetValuesMemOffSet(QUEUE queue, Tensor t, void *data, size_t bytes, size_t offset);

int TensorPutValuesMem(QUEUE queue, Tensor t, void *data, size_t bytes);

int TensorPutValuesMemOffSet(QUEUE queue, Tensor t, void *data, size_t bytes, size_t ofset);

int TensorGetNorm(QUEUE queue, Tensor t, double *norm);

void releaseTensor(Tensor *t);

void releaseTensorChar(TensorChar *t);


int dividirVetor(double *v, Tensor m, size_t len, double value, Kernel funcNorm, size_t max_works,
                 QUEUE queue);

int dividirVetorInt(unsigned char *src, double *dst, Tensor mi, Tensor mout, size_t len, double value,
                    Kernel funcNorm, size_t max_works, QUEUE queue);

int int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, Tensor mi, Tensor mout, size_t len, int nop,
                     Kernel func, QUEUE queue);

#endif //CNN_GPU_TENSOR_H
