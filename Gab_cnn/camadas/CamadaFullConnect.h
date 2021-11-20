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
	// funcao de ativacao e sua derivada
	int fa, dfa;
	Kernel fullfeed;
	Kernel fullCalcDWandFix;
	Kernel fullCalcDz;
	Kernel fullCalcDzandFixB;
	Kernel fullcalcin;
	RdP rdp_pesos;
	RdP rdp_bias;
} *CamadaFullConnect, CamadaFullConnect_t;

extern Camada CamadaFullConnect_new(Gpu gpu, Queue queue, P3d size_in, size_t tamanhoSaida,
									Tensor entrada, Parametros params, int funcaoDeAtivacao,
									Ecx exc, RdP rdp_pesos, RdP rdp_bias);


#endif //CNN_GPU_CAMADAFULLCONNECT_H
