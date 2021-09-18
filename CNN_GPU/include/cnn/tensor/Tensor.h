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
/**
 * Valido somente para tensores com host definido
 */
#define Tensor_Map_V(T, xx, yy, zz) ((double *)(T)->host)[(zz)*((T)->y*(T)->x)+(xx)*(T)->y+(yy)]

// TYPE_VALUE    COPY_PTR   TYPEMEMORY     OBSOLETO         3d/4d
// 000          	 0  		10            1               0

/// Tensor será armazenado na ram
#define TENSOR_RAM 0b00100
/// Tensor será armazenado na memoria compartilhada
#define TENSOR_SVM 0b01000
/// Tensor será armazenado no dispositivo(default)
#define TENSOR_GPU 0b00000

/// Será feita a copia do ponteiro passado
#define TENSOR_CPY 0b10000

/// Tensor 3D (defautl).
#define TENSOR3D 0b0
/// Tensor 4D.
#define TENSOR4D 0b1

/// Tensor tipo double (default).
#define TENSOR_DOUBLE 0x0
/// Tensor tipo char.
#define TENSOR_CHAR 0b00100000
/// Tensor tipo int.
#define TENSOR_INT 0b01000000

#define TENSOR_MASK_DIM         0x00000001
#define TENSOR_MASK_MEM         0b00001100
#define TENSOR_MASK_CPY         0b00010000
#define TENSOR_MASK_TYPE        0b01100000


typedef unsigned int UINT;
typedef unsigned int flag_t;
#define Ptr
#define var_host union {void *host;double *hostd;char *hostc;int *hosti;}

/**
 * Armazena as três dimensões de um ponto
 * @Atributtes x dimensão x do Ponto
 * @Atributtes y dimensão y do Ponto
 * @Atributtes z dimensão z do Ponto
 */
typedef struct Ponto {
	size_t x, y, z;
} Ponto;
typedef struct {
	int type;// -1 disable random,0 default, 1 uniform, 2 normal
	// y = X * a + b
	double a;
	double b;
} RandomParam;

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
 */
typedef struct Tensor_t {
	cl_mem data;
	UINT bytes;
	UINT x;
	UINT y;
	UINT z;
	UINT w;
	Ptr var_host;
	flag_t flag;
	cl_context context;
} *Tensor;
#define newTensor(context, queue, x, y, z, flag, error)new_Tensor(context,queue,flag,x,y,z,1,error,NULL)
#define newTensorChar(context, queue, x, y, z, flag, error)new_Tensor(context,queue,flag|TENSOR_CHAR,x,y,z,1,error,NULL)
#define newTensor4D(context, queue, x, y, z, w, flag, error)new_Tensor(context,queue,flag|TENSOR4D,x,y,z,w,error,NULL)

Tensor new_Tensor(cl_context context, QUEUE queue, char tensor_flag, UINT x, UINT y, UINT z, UINT w, CNN_ERROR *error,
				  void *p);

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

int TensorAt(Tensor t, UINT x, UINT y, UINT z, UINT w, UINT *index);

int TensorCpy(QUEUE queue, Tensor tdst, Tensor tsrc, size_t wsrc);

/***
 * v[i] = X * a + b
 * @param queue
 * @param t
 * @param distribuicao  "normal" para distribuição normal
 * @param a  fator de multiplicação, para distribuição normal é o desvio padrão
 * @param b  soma, para distribuição normal é a média
 * @return se falhar retorna um valor diferente de 0
 */
int TensorRandomize(QUEUE queue, Tensor t, int distribuicao, double a, double b);

void releaseTensor(Tensor *t);


#endif //CNN_GPU_TENSOR_H