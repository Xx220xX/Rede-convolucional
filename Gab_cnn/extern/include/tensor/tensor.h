//
// Created by hernique on 11/11/2021.
//
/***
 * Projeto tensor para uso de gpu e host
 * Fornece uma estrutura com diversas funções para manipular o tensor
 * Esse tensor Possui dimensão 4.
 *
 */

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "tensor_flags.h"
#include <time.h>

#include "config.h"
#include "exc.h"


#define TENSOR_NORMAL 2
#define TENSOR_UNIFORM 1

extern REAL (*Tensor_randn)();

extern REAL (*Tensor_rand)();

typedef unsigned char ubyte;
typedef struct {
	union {
		struct {

			ubyte dimensao4D: 1;
			ubyte _fill_0: 1;
			ubyte ram: 1;
			ubyte shared: 1;
			ubyte copy: 1;
			ubyte caractere: 1;
			ubyte inteiro: 1;
			ubyte _fill_1: 1;
		};
		ubyte flag;
	};

} TensorFlag __attribute__((aligned(1)));
typedef union {
	REAL *real;
	int *inteiro;
	ubyte *caractere;
	void *mem;
} Memory __attribute__((aligned(sizeof(void *))));
/**
 * Tensor
 */
typedef struct Tensor_t {
	/// parametros do tensor
	const TensorFlag flag;
	/// debug do tensor
	char *file_debug;

	/// dimensão do tensor
	size_t x, y, z, w,lenght;
	/// tamanho total de data
	size_t bytes;
	/// tamanho de um elemento
	unsigned int size_element;
	/// dados do tensor
	void *data;
	/// cl_command_queue para quando GPU ativo
	void *queue;
	/// cl_context para quando GPU ativo
	void *context;

	/// coloca os valores de data dentro do tensor, data deve ter o mesmo tamanho do tensor
	int (*setvalues)(struct Tensor_t *self, void *data);

	/// pega os valores de dentro do tensor para data, data deve ter o mesmo tamanho do tensor
	///(se data for nulo a funcao alocara os recursos que devaram ser liberados com free_mem)
	void *(*getvalues)(struct Tensor_t *self, void *data);

	/// coloca valores aleatorios no tensor, v[i] = X * a + b
	/// as funções Tensor_rand e Tensor_randn devem ser implementadas
	/// type = 2, Normal ; 1, Uniform
	int (*randomize)(struct Tensor_t *self, int type, REAL a, REAL b);

	/// modifica os nbytes  do tensor com o offset aplicado
	///  v[offset:offset+bytes] = data
	int (*setvaluesM)(struct Tensor_t *self, size_t offset, void *data, size_t n_bytes);

	/// devolve os nbytes  do tensor com o offset aplicado
	/// data = v[offset:offset+bytes]
	/// (se data for nulo a funcao alocara os recursos que devaram ser liberados com free_mem)
	void *(*getvaluesM)(struct Tensor_t *self, size_t offset, void *data, size_t n_bytes);

	/// preenche o tensor com o valor
	int (*fill)(struct Tensor_t *self,char partern);
	/// preenche a região com um valor
	int (*fillM)(struct Tensor_t *self,size_t offset,size_t bytes,void *patern, size_t size_patern);

	/// retorna uma string(deve ser liberados os recursos com free_mem) contendo o tensor no formato CamadaConv_json
	char *(*json)(struct Tensor_t *self, int showValues);

	///  libera os recursos internos do tensor
	void (*release)(struct Tensor_t **self_p);

	/// debug do tensor
	void (*registreError)(struct Tensor_t *self, char *format, ...);

	/// coloca o tensor em uma imagem cinza , imagem[i0:h,j0:w] = tensor[:,:,z,l]
	int (*imagegray)(struct Tensor_t *self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l);

	/// ponteiro utilizado para debugar stack
	/// controle de erros internos
	Ecx erro;
} *Tensor, Tensor_t;

/***
 *  Cria um tensor alocado
 *  [w0[z0[x0[y0, y1 ...],x1 ... ], z1 ... ], w1 ...]
 *  da menor dimensão para maior
 *  y -> x -> z - > w
 *
 * @param x dimensão x do tensor
 * @param y dimensão y do tensor
 * @param z dimensão z do tensor
 * @param w dimensão w do tensor
 * @param flag caracteristicas do tensor (4D,3D),(RAM,SVM,GPU),(INT,CHAR,REAL)
 * @param ... caso GPU esteja setado, deve-se enviar o contexto e a queue,
 * 			esta deve existir enquanto o tensor for utilizado.
 * 			Caso copy esteja ativado o ultimo parametro deve ser um ponteiro com o mesmo tamanho dos dados do tensor
 *
 * @return Tensor
 */
Tensor Tensor_new(size_t x, size_t y, size_t z, size_t w, Ecx ecx, int flag, ...);


#endif //TENSOR_TENSOR_H
