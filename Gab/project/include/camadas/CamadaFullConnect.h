//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADAFULLCONNECT_H
#define CNN_GPU_CAMADAFULLCONNECT_H

#include"camada.h"

#include"funcoesDeAtivacao.h"


typedef struct CamadaFullConnect_t {
	Camada_t super;
	Tensor w;
	Tensor dw;
	Tensor b;
	Tensor db;
	Tensor z;
	Tensor dz;
	Tensor daf;
	Tensor expoente;
	REAL *values;
	REAL soma;
	REAL maximo;
	// funcao de ativacao e sua derivada
	FAtivacao fa;
	uint32_t  dfa;
	uint32_t  flatten;

	cl_kernel feed;
	cl_kernel ativa;
	cl_kernel calc_exp;
	cl_kernel calc_dzdb;
	cl_kernel calc_dzdb_batch;
	cl_kernel corrige_peso;
	cl_kernel calc_da;
	cl_kernel calc_dw;
	cl_kernel calc_dw_batch;

	RandomParams rdp_pesos;
	RandomParams rdp_bias;
} *CamadaFullConnect, CamadaFullConnect_t;

extern Camada CamadaFullConnect_new(Gpu gpu, Queue queue, P3d size_in, size_t tamanhoSaida, Tensor entrada, Parametros params, FAtivacao_t funcaoDeAtivacao, Ecx exc, RandomParams rdp_pesos, RandomParams rdp_bias);

extern Camada CamadaFullConnect_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADAFULLCONNECT_H
