//
// Created by Henrique on 21/11/2021.
//

#ifndef GAB_CNN_CNN_H
#define GAB_CNN_CNN_H

#include <tensor/tensor.h>
#include <gpu/Kernel.h>
#include <camadas/camada.h>
#include "lcg/lcg.h"


typedef struct Cnn_t {
	///  versão da compilação
	const char *version;

	/// entrada da rede
	Tensor entrada;
	/// Ultimo gradiente
	Tensor ds;
	/// Tensor de objetivo da rede
	Tensor target;
	/// numero de camadas
	size_t l;
	/// camadas
	Camada *cm;
	/// kernels internos
	void *kernels;
	/// Lua vm
	void *LuaVm;

	void (*releaseL)(void *L);

	/// entrada da rede
	const P3d size_in;
	Ecx ecx;

	Gpu gpu;
	int8_t release_gpu;
	Queue queue;

	// methods
	int (*setInput)(struct Cnn_t *self, size_t x, size_t y, size_t z);

	int (*release)(struct Cnn_t **selfp);

	/// retorna a dimensão da saída da rede
	P3d (*getSizeOut)(struct Cnn_t *self);

	/// retorna a rede em json
	char *(*json)(struct Cnn_t *self, int showValue);

	void (*jsonF)(struct Cnn_t *self, int showValue, const char *fileName);

	int (*save)(struct Cnn_t *self, const char *filename);

	int (*load)(struct Cnn_t *self, const char *filename);

	int (*predict)(struct Cnn_t *self, Tensor input);

	int (*predictv)(struct Cnn_t *self, REAL *input);

	int (*learn)(struct Cnn_t *self, Tensor target);

	int (*learnv)(struct Cnn_t *self, REAL *target);

	REAL (*mse)(struct Cnn_t *self);

	REAL (*mseT)(struct Cnn_t *self, Tensor target);

	int (*maxIndex)(struct Cnn_t *self);

	void (*print)(struct Cnn_t *self, const char *comment);

	void (*fprint)(struct Cnn_t *self, FILE *fdst, const char *comment);

	char *(*printstr)(struct Cnn_t *self, const char *comment);

	int (*normalizeIMAGE)(struct Cnn_t *self, Tensor dst_real, Tensor src_char);

	int (*extractVectorLabelClass)(struct Cnn_t *self, Tensor dst, Tensor label);

	int (*Convolucao)(struct Cnn_t *self, P2d passo, P3d filtro, Parametros p, RandomParams filtros);

	int (*ConvolucaoF)(struct Cnn_t *self, P2d passo, P3d filtro, uint32_t funcaoAtivacao, Parametros p, RandomParams filtros);

	int (*ConvolucaoNC)(struct Cnn_t *self, P2d passo, P2d abertura, P3d filtro, uint32_t funcaoAtivacao, Parametros p, RandomParams filtros);

	int (*Pooling)(struct Cnn_t *self, P2d passo, P2d filtro, uint32_t type);

	int (*Relu)(struct Cnn_t *self, REAL fator_menor0, REAL fator_maior0);

	int (*PRelu)(struct Cnn_t *self, Parametros params, RandomParams rdp_a);

	int (*FullConnect)(struct Cnn_t *self, size_t numero_neuronios, Parametros p, uint32_t funcaoAtivacao, RandomParams rdp_pesos, RandomParams rdp_bias);

	int (*Padding)(struct Cnn_t *self, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);

	int (*DropOut)(struct Cnn_t *self, REAL probabilidadeSaida, cl_ulong seed);

	int (*SoftMax)(struct Cnn_t *self, int8_t flag);

	int (*BatchNorm)(struct Cnn_t *self, size_t batch_size, REAL epsilon, Parametros p, RandomParams randY, RandomParams randB);

	void (*removeLastLayer)(struct Cnn_t *self);
} *Cnn, Cnn_t;

extern const char *Cnn_version();

extern Cnn Cnn_new();

#endif //GAB_CNN_CNN_H
