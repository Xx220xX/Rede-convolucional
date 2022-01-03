//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADADROPOUT_H
#define CNN_GPU_CAMADADROPOUT_H

#include"camada.h"


typedef struct CamadaDropOut_t {
	/// herda os atributos da classe mãe Camada
	Camada_t super;
	/// armazena as posições que foram sorteadas para a saída
	Tensor hitmap;
	/// probabilidade de saída
	REAL probabilidade_saida;
	/// semente para utilizar o algoritmo pseudo aleatorio
	cl_ulong seed;
	/***
 * niters = tamanho da saída
 * A iteração é feita no tensor de saída
 * @param entrada tensor de entrada(leitura)
 * @param saida  tensor de saída (escrita)
 * @param hitmap tensor de mapeamento (char)(escrita)
 * @param seed semente, a nova semente deve ser atualizada por  seed = ((seed+niters) * 0x5deece66dULL + 0xbULL) & ((1ULL << 31) - 1);
 * @param pativa probabilidade do neuronio ser ativado
 * @param k0 uso interno do kernel
 */
	Kernel dropativa;
	/***
 * A iteração é feita no gradiente de entrada
 * @param gradentrada (escrita)
 * @param hitmap (leitura)
 * @param gradnext (leitura)
 * @param k0 uso interno do kernel
 */
	Kernel dropcalcgrad;
} *CamadaDropOut, CamadaDropOut_t;


extern Camada CamadaDropOut_new(Gpu gpu, Queue queue, P3d size_in, REAL probabilidade_saida, cl_ulong seed, Tensor entrada, Ecx ecx);

extern Camada CamadaDropOut_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADADROPOUT_H
