//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONVF_H
#define CNN_GPU_CAMADACONVF_H

#include "string.h"
#include"camada.h"
#include "funcoesDeAtivacao.h"

typedef struct CamadaConvF_t {
	Camada_t super;
	Tensor w;
	Tensor dw;
	Tensor z;
	Tensor dz;
	Tensor b;
	Tensor db;
	size_t passox, passoy;
	cl_kernel conv;
	cl_kernel calc_dz;
	cl_kernel calc_da;
	cl_kernel calc_dw;
	cl_kernel calc_dw_batch;
	cl_kernel calc_db;
	cl_kernel calc_db_batch;
	cl_kernel corrige_peso;
	size_t pad_top, pad_bottom, pad_left, pad_right;
	Kernel convFSum;
	Kernel convFCalcGradZ;
	Kernel convFCalcGradBAndFix;
	Kernel convFCalcGradAndFixWeight;
	Kernel convFCalcGradIn;
	Kernel convFCalcGradBBatch;
	Kernel convFCalcGradBatch;
	Kernel kernel_fixW;
	FAtivacao fa;
	uint32_t derivationFuntion;
	RandomParams rdp_filtros;
} *CamadaConvF, CamadaConvF_t;

extern Camada CamadaConvF_new(INTERNAL_DEFAULT_ARGS, P2d passo, P3d filtro, FAtivacao_t ativacao, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right,Parametros params, RandomParams rdp_filtros);

extern Camada CamadaConvF_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);

#endif //CNN_GPU_CAMADACONVF_H
