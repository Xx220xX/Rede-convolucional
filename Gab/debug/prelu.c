//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "matlab.h"
#include "camadas/CamadaRelu.h"
#include<string.h>
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)


int main() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(30, 40, 8, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->PRelu(cnn, Params(1e-3), RDP(0));
	P3d  sizeout = cnn->getSizeOut(cnn);
	Tensor target = Tensor_new(unP3D(sizeout), 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	CamadaPRelu cf = CST_PRELU(cnn,0);
	entrada->randomize(entrada,TENSOR_NORMAL,1,0);
	target->randomize(target,TENSOR_NORMAL,1,0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->predict(cnn, entrada);
	FILE *f = fopen("D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\preluu.m", "w");
	matlab("clc;close all;clear all");
	_tomatlab(cf->super.s, f, "s","mshape");
	_tomatlab(cf->super.a, f, "a","mshape");
	_tomatlab(target, f, "t","mshape");
	_tomatlab(cf->A,f,"A","mshape");
	matlab("sm = (a>0).*a + (a<0).*(a.*A);");

	cnn->learn(cnn, target);
	_tomatlab(cnn->ds, f, "ds", "mshape");
	_tomatlab(cf->super.da, f, "da","mshape");

	matlab("dsm = sm - t;");
	matlab("dam = ((a>0)*1 + (a<0).*A).*dsm;");
	_tomatlab(cf->super.da,f,"da","mshape");
	matlabCmp("s","sm");
	matlabCmp("ds","dsm");
	matlabCmp("da","dam");

	fclose(f);
	Release(entrada);
	Release(target);

	system("start D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\" );
	return cnn->release(&cnn);
}