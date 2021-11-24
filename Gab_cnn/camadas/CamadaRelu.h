//
// Created by gf154 on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_RELU_H
#define CNN_GPU_CAMADA_RELU_H

#include "camada.h"

typedef struct CamadaRelu_t {
	/// herda os atributos da classe mãe Camada
	Camada_t super;
	/// fator de multiplicação para valores menores que 0
	REAL lessoh;
	/// fator de multiplicação para valores maiores que 0
	REAL greateroh;
	/***
 * Itera na saída
 * @param entrada tensor de entrada (leitura)
 * @param saida tensor de saída (escrita)
 * @param menor (contante de multiplicação para valores menores que 0
 * @param maior (contante de multiplicação para valores maiores que 0
 * @param k0 uso interno do kernel
 */
	Kernel reluativa;
	/***
 * Itera na entrada
 * @param gradentrada  tensor de gadiente de entrada (escrita)
 * @param entrada tensor de entrada (leitura)
 * @param gradnext tensor gradiente da saída (leitura)
 * @param menor (contante de multiplicação para valores menores que 0
 * @param maior (contante de multiplicação para valores maiores que 0
 * @param k0 uso interno do kernel
 */
	Kernel relucalcgrad;
} *CamadaRelu, CamadaRelu_t;


Camada CamadaRelu_new(Gpu gpu, Queue queue, P3d size_in, REAL less, REAL greater, Tensor entrada,
					  Ecx ecx);

extern Camada CamadaRelu_load(FILE *f, Gpu gpu, Queue queue, P3d sizein, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADA_RELU_H
