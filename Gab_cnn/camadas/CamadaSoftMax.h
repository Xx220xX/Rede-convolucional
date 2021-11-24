//
// Created by Henrique on 5/8/2021.
//

#ifndef CNN_GPU_CAMADASOFTMAX_H
#define CNN_GPU_CAMADASOFTMAX_H

#include "camada.h"


typedef struct CamadaSoftMax_t {
	/// Classe mãe
	Camada_t super;
	/// armazena a soma das exponenciais
	Tensor soma;
	/// armazena a exponencial da entrada
	Tensor exponent;
	/**
 * @goal calcular e^a(x,y,z)
 * @iteration dimensão de a (x,y,z)
 * @param entrada Tensor de entrada (leitura)
 * @param exponent Tensor e^entrada (escrita)
 * @param k0 usado internamente no kernel
 */
	Kernel softmaxExp;
	/***
 * @goal encontrar a soma de cada dimensão z
 * @iteration dimensão z da entrada a(:,:,z)
 * @param eps Tensor exponencial da entrada (leitura)
 * @param soma Tensor da soma das exponenciais (escrita)
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão x da saída
 * @param k0 usado internamente no kernel
 */
	Kernel softmaxSomaExp;
	/***
 * @goal Normalizar a exponencial pela soma
 *  * @iteration dimensão da saída  s(x,y,z)
 * @param exponet Tensor exponencial da entrada (leitura)
 * @param soma Tensor da soma das exponenciais (leitura)
 * @param saida Tensor de saída (escrita)
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão x da saída
 * @param k0 usado internamente no kernel
 */
	Kernel softmaxNormaliza;
	/**
 * @goal Calcular os gradientes de entrada
 * @iteration dimensão da entrada a(x,y,z)
 * @param gradentrada Tensor de gradientes de entrada (escrita)
 * @param entrada Tensor de entrada (leitura)
 * @param gradnext Tensor gradiente da saída (leitura)
 * @param k0 usado internamente no kernel
 */
	Kernel softMaxcalcgrad;
} *CamadaSoftMax, CamadaSoftMax_t;

extern Camada CamadaSoftMax_new(Gpu gpu, Queue queue, P3d size_in, Tensor entrada, Ecx ecx);
extern Camada CamadaSoftMax_load(FILE *f, Gpu gpu, Queue queue, P3d sizein, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADASOFTMAX_H
