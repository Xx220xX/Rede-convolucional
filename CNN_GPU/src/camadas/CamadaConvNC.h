
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


typedef unsigned int UINT;


typedef struct {
	Typecamada super;
	Tensor filtros;
	Tensor grad_filtros;
	Tensor grad_filtros_old;
	UINT passox,largx,passoy,largy, numeroFiltros;
	Kernel kernelConvNcSum;
	Kernel kernelConvNcFixWeight;
	Kernel kernelConvNcCalcGradsFiltro;
	Kernel kernelConvNcCalcGrads;
} *CamadaConvNc, TypecamadaConvNc;

void calc_gradsConvNc(CamadaConvNc c, Tensor Gradnext);

void releaseConvNc(CamadaConvNc *pc);

int ativaConvNc(CamadaConvNc c);

void corrige_pesosConvNc(CamadaConvNc c);

int ConvNcRandomize(CamadaConvNc c, WrapperCL *cl, GPU_ERROR *error);

void salvarConvNc(WrapperCL *cl, CamadaConvNc c, FILE *dst, GPU_ERROR *error);

Camada createConvNc(WrapperCL *cl, cl_command_queue queue, UINT passox,
                    UINT passoy, UINT largx, UINT largy, UINT filtrox, UINT filtroy,
                    UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                    Tensor entrada, Params *params, GPU_ERROR *error, int randomize);


#endif //CNN_GPU_CAMADAConvNc_H
