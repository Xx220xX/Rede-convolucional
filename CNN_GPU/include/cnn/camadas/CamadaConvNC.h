
//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADAConvNc_H
#define CNN_GPU_CAMADAConvNc_H

#include "string.h"
#include"Camada.h"
#include"../tensor/Tensor.h"
#include <stdlib.h>
#include"utils.h"


typedef struct {
	Typecamada super;
	Tensor filtros;
	Tensor grad_filtros;
	Tensor grad_filtros_old;
	UINT passox;
	UINT passoy;
	UINT largx;
	UINT largy;
	UINT numeroFiltros;
	Kernel kernelConvNcSum;
	Kernel kernelConvNcFixWeight;
	Kernel kernelConvNcCalcGradsFiltro;
	Kernel kernelConvNcCalcGrads;
} *CamadaConvNc, TypecamadaConvNc;



Camada createConvNc(WrapperCL *cl, QUEUE queue, UINT passox,
					UINT passoy, UINT largx, UINT largy, UINT filtrox, UINT filtroy,
					UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
					Tensor entrada, Params params, char usehost, CNN_ERROR *error, int randomize);


#endif //CNN_GPU_CAMADAConvNc_H
