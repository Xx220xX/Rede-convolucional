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
	/// armazena os maximos
	Tensor maximos;
	/// armazena o indices dos maximos
	Tensor indice_maximos;
	/// armazena a exponencial da entrada
	Tensor exponent;
	/// Tensor para armazenar a derivada da camada softmax
	Tensor ds;

	/// flag para identificar qual tipo de camada
	const char flag;
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
 * @param da Tensor de gradientes de entrada (escrita)
 * @param s Tensor de saida (leitura)
 * @param ds Tensor gradiente da saída (leitura)
 * @param sx dimensão x da saída
 * @param sy dimensão y da saída
 * @param k0 usado internamente no kernel
 */
	Kernel softMaxcalcgrad;
	/**
* @goal Calcular os gradientes de entrada
* @iteration dimensão da entrada a(x,y,z)
* @param da Tensor de gradientes de entrada (escrita)
* @param ds Tensor gradiente da saída (leitura)
* @param sx dimensão x da saída
* @param sy dimensão y da saída
* @param k0 usado internamente no kernel
*/
	Kernel softMaxcalcgradWhenNorm;
	Kernel softmaxFindMax;
} *CamadaSoftMax, CamadaSoftMax_t;

extern Camada CamadaSoftMax_new(Gpu gpu, Queue queue, char flag, P3d size_in, Tensor entrada, Ecx ecx);

extern Camada CamadaSoftMax_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADASOFTMAX_H
