//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADADROPOUT_H
#define CNN_GPU_CAMADADROPOUT_H

#include "string.h"
#include"Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>
#include <float.h>


typedef struct {
	Typecamada super;
	TensorChar hitmap;
	char flag_releaseInput;
	double p_ativacao;
	cl_ulong seed;
	Kernel kerneldropativa;
	Kernel kerneldropcalcgrad;
} *CamadaDropOut, Typecamadadropout;

void releaseDropOut(CamadaDropOut *pc);

int corrigePesosDropOut(CamadaDropOut c);

int ativaDropOut(CamadaDropOut c);

int calc_gradsDropOut(CamadaDropOut c, Tensor GradNext);

void salvarDropOut(WrapperCL *cl, CamadaDropOut c, FILE *dst, GPU_ERROR *error);

Camada createDropOut(WrapperCL *cl,QUEUE queue, UINT inx, UINT iny, UINT inz, double p_ativacao, long long seed, Tensor entrada,
                     GPU_ERROR *error);

void releaseDropOut(CamadaDropOut *pc) ;

int ativaDropOut(CamadaDropOut c);

int corrigePesosDropOut(CamadaDropOut c) ;

int calc_gradsDropOut(CamadaDropOut c, Tensor GradNext);


void salvarDropOut(WrapperCL *cl, CamadaDropOut c, FILE *dst, GPU_ERROR *error);

Camada carregarDropOut(WrapperCL *cl, FILE *src,QUEUE queue, Tensor entrada,
					   Params params, GPU_ERROR *error) ;

#endif //CNN_GPU_CAMADADROPOUT_H
