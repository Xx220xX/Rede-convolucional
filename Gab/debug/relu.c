//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "camadas/CamadaRelu.h"
#include<string.h>
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define matlabf(format, ...) fprintf(f,format,##__VA_ARGS__);fprintf(f,"\n")
#define matlab(x) fprintf(f,"%s\n",x)

#define matlabCmp(x, y)matlab("figure;hold on;"); \
matlab("plot("x"(:),'DisplayName','"x"')");\
matlab("plot("y"(:),'DisplayName','"y"')");      \
matlab("title('"#x" vs "#y"');");\
matlab("legend();");                                  \
matlab("if gcf() == 1");\
matlabf("print('%s.pdf')",__FILENAME__);                     \
matlabf("else print('%s.pdf', '-append'); end;",__FILENAME__)


int main() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(30, 40, 8, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->Relu(cnn,0.2,1);
	P3d  sizeout = cnn->getSizeOut(cnn);
	Tensor target = Tensor_new(unP3D(sizeout), 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	CamadaRelu cf = (CamadaRelu) cnn->cm[0];
	entrada->randomize(entrada,TENSOR_NORMAL,1,0);
	target->randomize(target,TENSOR_NORMAL,1,0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->predict(cnn, entrada);
	FILE *f = fopen("D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\reluu.m", "w");
	matlab("clc;close all;clear all");
	_tomatlab(cf->super.s, f, "s","mshape");
	_tomatlab(cf->super.a, f, "a","mshape");
	_tomatlab(target, f, "t","mshape");

	cnn->learn(cnn, target);
	_tomatlab(cnn->ds, f, "ds", "mshape");
	_tomatlab(cf->super.da, f, "da","mshape");

	matlabf("sm =  a.*((a>0)*%lf + (a<0)*%lf);",cf->greateroh,cf->lessoh);
	matlab("dsm = sm - t;");
	_tomatlab(cf->super.da,f,"da","mshape");
	matlabf("dam =  dsm.*((a>0)*%lf + (a<0)*%lf);",cf->greateroh,cf->lessoh);
	matlabCmp("s","sm");
	matlabCmp("ds","dsm");
	matlabCmp("da","dam");

	fclose(f);
	Release(entrada);
	Release(target);

//	system("start D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\" );
	return cnn->release(&cnn);
}