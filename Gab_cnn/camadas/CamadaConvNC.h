
//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADAConvNc_H
#define CNN_GPU_CAMADAConvNc_H

#include"camada.h"


typedef struct CamadaConvNc_t {
	Camada_t super;
	Tensor filtros;
	Tensor grad_filtros;
	Tensor grad_filtros_old;
	size_t passox, passoy;
	size_t largx, largy;
	size_t numeroFiltros;
	Kernel kernelConvNcSum;
	Kernel kernelConvNcFixWeight;
	Kernel kernelConvNcCalcGradsFiltro;
	Kernel kernelConvNcCalcGrads;
} *CamadaConvNc, CamadaConvNc_t;


extern Camada createConvNc(Gpu gpu, QUEUE queue, P3d  passo, P3d larg, P3d filtro, P3d size_in, Tensor entrada, Params params, RandomParam randomParams);


#endif //CNN_GPU_CAMADAConvNc_H
