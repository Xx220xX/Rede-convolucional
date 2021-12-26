//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "camadas/all_camadas.h"
#include "matlab.h"
#include<string.h>


int main() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(20, 20, 31, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->BatchNorm(cnn, 20, 1e-12, Params(1e-3,0), RDP(0), RDP(0));
	P3d sizeout = cnn->getSizeOut(cnn);
	Tensor target = Tensor_new(unP3D(sizeout), 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	CamadaBatchNorm cf = CST_BATCHNORM(cnn, 0);
	entrada->randomize(entrada, TENSOR_GAUSSIAN, 1, 0);
	target->randomize(target, TENSOR_GAUSSIAN, 1, 0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	for (int i = 0; i < 300; ++i) {
		cnn->predict(cnn, entrada);
		cnn->learn(cnn, target);
		VAR(cf->super.da);
		LN();
	}
	Release(entrada);
	Release(target);


	return cnn->release(&cnn);

}