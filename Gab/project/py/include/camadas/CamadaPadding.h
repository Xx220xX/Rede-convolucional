//
// Created by Henrique on 03/08/2021.
//

#ifndef CNN_GPU_CAMADA_Padding_H
#define CNN_GPU_CAMADA_Padding_H


#include "camada.h"

typedef struct CamadaPadding_t {
	Camada_t super;
	uint32_t top, bottom, left, right;
	/**
 * Iterado pela dimensão da entrada
 * @param in Tensor de entrada
 * @param out Tensor de saída
 * @param txi dimensão x da entrada
 * @param tyi dimensão y da entrada
 * @param txo dimensão x da saída
 * @param tyo dimensão y da saída
 * @param t  top pad
 * @param l left pad
 * @param k0 usado para o lancamento do kernel
 */
	Kernel paddingfeed;
	/**
 * Iterado pela dimensão da entrada
 * @param gradNext gradiente da saída (leitura)
 * @param gradin gradiente da entrada (escrita)
 * @param txi dimensão x entrada
 * @param tyi dimensão y entrada
 * @param txo dimensão x saída
 * @param tyo dimensão y saída
 * @param t top pad
 * @param l left pad
 * @param k0 usado para lançamento do kernel
 */
	Kernel paddingBack;
} *CamadaPadding, CamadaPadding_t;

extern Camada CamadaPadding_new(Gpu gpu, Queue queue, P3d size_in, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right,
								Tensor entrada, Ecx ecx);

extern Camada CamadaPadding_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADA_Padding_H
