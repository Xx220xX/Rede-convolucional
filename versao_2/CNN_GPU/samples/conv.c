//
// Created by Henrique on 07/11/2021.
//
/**
 * Este programa faz os teste para uma camada convolucional
 * @return
 */
#include "utils/manageTrain.h"
#include "utils/defaultkernel.h"
#include <time.h>

#define  F ConvolucaoF(c,1,1,2,2,2,FTANH,(RandomParam){0});CamadaConvF  cv = (CamadaConvF) c->camadas[c->size - 1];Tensor s = cv->super.saida;
#define  C Convolucao(c,1,1,2,2,2,(RandomParam){0});CamadaConv  cv = (CamadaConv) c->camadas[c->size - 1];Tensor s = cv->super.saida;

int main(){
	LCG_setSeed(515);
	Cnn c = createCnnWithWrapperProgram(default_kernel,(Params){0.1,0,0},3,3,1,CL_DEVICE_TYPE_GPU);
	F
	/*
	 * {[(0.83, 0.38, -0.30)
(0.99, -0.80, 0.44)
(0.71, 0.48, 0.53)]}

{[(-0.03, 0.11)
(0.06, 0.08)]}
{[(0.05, -0.14)
(-0.04, 0.18)]}

{[(0.01, -0.05)
(-0.04, 0.14)]
[(-0.20, 0.17)
(0.22, -0.02)]}
	 */




	Tensor entrada = newTensor(c->cl->context,c->queue,3,3,1,0,&c->error);
	TensorRandomize(c->queue,entrada,LCG_UNIFORM,2,-1);


	printTensor(c->queue,entrada,stdout);
	printTensor(c->queue,cv->filtros,stdout);

	CnnCallT(c,entrada);
	synchronizeKernel(c->queue);
	printTensor(c->queue,s,stdout);


	releaseCnn(&c);
	releaseTensor(&entrada);
	return 0;
}