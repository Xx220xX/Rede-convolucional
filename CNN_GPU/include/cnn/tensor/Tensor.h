//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_TENSOR_H
#define CNN_GPU_TENSOR_H

#include <stdlib.h>
#include <math.h>
#include "config.h"
#include"../gpu/WrapperCL.h"
#include "../camadas/utils.h"
#include"../cnn_errors_list.h"

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

/// USE DRIVE. Alloca recursos no drive (default).
#define TENSOR_DRIV 0b00000
/// USE host.  Os recursos serão alocados apenas no host
#define TENSOR_HOST 0b00010

/// valido somente para quando TENSOR_DRIV estiver ativo
#define TENSOR_NCPY 0b00000

/// será utilizado um ponteiro para criar a memoria no drive.
#define TENSOR_UPTR 0b00100
/// será feito o uso da memoria compartilhada
#define TENSOR_SMEM 0b01000
/// será feito o uso da memoria comum
#define TENSOR_HMEM 0b10000


/// Tensor 3D (defautl).
#define TENSOR3D 0b0
/// Tensor 4D.
#define TENSOR4D 0b1

/// Tensor tipo double (default).
#define TENSOR_DOUBLE 0x0
/// Tensor tipo char.
#define TENSOR_CHAR 0x10
/// Tensor tipo int.
#define TENSOR_INT 0x20

#define TENSOR_MASK_DIM         0x00001
#define TENSOR_MASK_MEM         0b11100
#define TENSOR_MASK_DRIVEORHOST 0b00010

#define TENSOR_MASK_TYPE 0x30

typedef unsigned int UINT;
typedef unsigned int flag_t;

/**
 * Armazena as três dimensões de um ponto
 * @Atributtes x dimensão x do Ponto
 * @Atributtes y dimensão y do Ponto
 * @Atributtes z dimensão z do Ponto
 */
typedef struct Ponto {
	size_t x, y, z;
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
	flag_t flag;

	cl_context context;

} *Tensor;
#define newTensor(context,queue,x,y,z,flag,error)new_Tensor(context,queue,flag,x,y,z,1,error,NULL)
#define newTensorChar(context,queue,x,y,z,flag,error)new_Tensor(context,queue,flag|TENSOR_CHAR,x,y,z,1,error,NULL)
#define newTensor4D(context,queue,x,y,z,w,flag,error)new_Tensor(context,queue,flag|TENSOR4D,x,y,z,w,error,NULL)

Tensor new_Tensor(cl_context context, QUEUE queue, char tensor_flag, UINT x, UINT y, UINT z, UINT w, Exception *error, void *p);

void printTensor(QUEUE q, Tensor t, FILE *f);

int TensorFill(QUEUE queue, Tensor t, char patern);

int TensorFillOffSet(QUEUE queue, Tensor t, char patern, size_t offset);

int TensorFillDouble(QUEUE queue, Tensor t, double pattern);

int TensorFillDoubleOffSet(QUEUE queue, Tensor t, double pattern, size_t offset);

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



int dividirVetor(double *v, Tensor m, size_t len, double value, Kernel funcNorm, size_t max_works,
                 QUEUE queue);


#endif //CNN_GPU_TENSOR_H
