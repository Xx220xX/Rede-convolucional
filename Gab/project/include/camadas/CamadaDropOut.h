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
	Kernel dropativa;
	Kernel dropcalcgrad;

	void (*setMode)(struct CamadaDropOut_t *self, int training);
} *CamadaDropOut, CamadaDropOut_t;


extern Camada CamadaDropOut_new(Gpu gpu, Queue queue, P3d size_in, REAL probabilidade_saida, cl_ulong seed, Tensor entrada, Ecx ecx);

extern Camada CamadaDropOut_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);


#endif //CNN_GPU_CAMADADROPOUT_H
