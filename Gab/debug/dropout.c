//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "matlab.h"
#include<string.h>


int main() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(1, 10, 1, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->DropOut(cnn, 0.7, time(0));
	P3d sizeout = cnn->getSizeOut(cnn);
	Tensor target = Tensor_new(unP3D(sizeout), 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	CamadaDropOut cf = CST_DROPOUT(cnn, 0);
	entrada->randomize(entrada, TENSOR_GAUSSIAN, 1, 0);
	target->randomize(target, TENSOR_GAUSSIAN, 1, 0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	matlabInit();
	cnn->predict(cnn, entrada);
	Trmatlab(cf->hitmap,"h");
	Trmatlab(cf->super.a,"a");
	Trmatlab(cf->super.s,"s");
	cnn->learn(cnn, target);
	Trmatlab(cnn->ds,"ds");
	Trmatlab(cf->super.da,"da");
	matlab("sum(abs(h(:)))/length(h(:))");
	matlab("sm = a .* h;");
	matlabCmp("s","sm");
	Release(entrada);
	Release(target);
	matlabEnd();

	return cnn->release(&cnn);

}