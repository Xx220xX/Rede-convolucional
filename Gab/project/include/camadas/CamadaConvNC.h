//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADAConvNc_H
#define CNN_GPU_CAMADAConvNc_H

#include"camada.h"


typedef struct CamadaConvNC_t {
	/// classe mãe
	Camada_t super;
	/// filtros
	Tensor W;
	/// gradiente dos filtros
	Tensor dW;
	/// soma convolucional
	Tensor z;
	/// gradiente da soma convolucional
	Tensor dz;
	/// passo
	size_t passox, passoy;
	/// largura da abertura
	size_t aberturax, aberturay;
	/// função de ativação
	uint32_t activationFuntion;
	/// derivada da função de ativação
	uint32_t derivationFunction;
	/// geração aleatória dos filtros
	RandomParams rdp_filtros;
	Kernel convncSum;
	Kernel convncCalcGradZ;
	Kernel convncCalcFiltro;
	Kernel convncCalcGrads;
	Kernel convncCalcFiltroBatch;
	Kernel kernel_fixW;

} *CamadaConvNC, CamadaConvNC_t;


extern Camada CamadaConvNC_new(Gpu gpu, Queue queue, P2d passo, P2d abertura, P3d filtro, P3d size_in, uint32_t ativacao, Tensor entrada, Parametros params, Ecx ecx, RandomParams rdp_filtros);

extern Camada CamadaConvNC_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx);


#endif //CNN_GPU_CAMADAConvNc_H
