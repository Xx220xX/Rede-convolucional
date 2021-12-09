//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn_lua.h"
#include <camadas/CamadaSoftMax.h>
#include "lcg/lcg.h"
#include "string.h"


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
#define _tomatlabz_xy(tensor, name)\
_tomatlab(tensor, f, name, NULL);\
matlabf(name" = mshape("name",[%zu,%zu]);", tensor->z,tensor->x * tensor->y)

int main() {
	Cnn cnn = Cnn_new();
	LCG_setSeed(time(0));
	Tensor entrada = Tensor_new(2, 1, 12, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->SoftMax(cnn, SOFTLAST | SOFTNORM);

	Tensor target = Tensor_new(unP3D(cnn->getSizeOut(cnn)), 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	CamadaSoftMax cf = (CamadaSoftMax) cnn->cm[0];
	entrada->randomize(entrada, TENSOR_NORMAL, 1, 0);
	target->randomize(target, TENSOR_NORMAL, 1, 0);

	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
		cnn->predict(cnn, entrada);
	FILE *f = fopen("D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\softMax.m", "w");
	matlab("clc;close all;clear all");
	_tomatlabz_xy(cf->super.a, "a");
	if (cf->flag & SOFTNORM) {
		_tomatlabz_xy(cf->maximos, "mx");
		_tomatlabz_xy(cf->indice_maximos, "imx");

	}
	_tomatlabz_xy(cf->exponent, "ex");
	_tomatlabz_xy(cf->soma, "soma");

	_tomatlabz_xy(cf->super.s, "s");

	if (cf->flag & SOFTNORM) {
		matlab("mxm = max(a')';\n"
			   "imxm = find(a'==mxm') - [1:size(a,2):size(a,2)*size(a,1)]';\n"
			   "exm = exp(a-mxm);");
		matlabCmp("mx", "mxm");
		matlabCmp("imx", "imxm");

	} else {
		matlab("exm = exp(a);");
	}
	matlab("somam = sum(exm')';"
		   "sm = exm./somam;\n");

	cnn->learn(cnn, target);
	_tomatlabz_xy(target, "t");
	_tomatlabz_xy(cnn->ds, "ds");
	_tomatlabz_xy(cf->super.da, "da");
	matlabCmp("ex", "exm");
	matlabCmp("soma", "somam");
	matlabCmp("s", "sm");
	matlab("dsm = sm - t;");
	if (cf->flag & SOFTLAST) {
		if (cf->flag & SOFTNORM) {
			matlab("tmp = reshape([0:size(dsm,2)^2-1],size(dsm,2),size(dsm,2))';\n"
				   "i = floor(tmp/size(a,2));\n"
				   "j = mod(tmp,size(a,2));\n"
				   "tmp = (i==j)-(j==reshape(imxm,[1,1,size(imxm,1)]));\n"
				   "grad = dsm;\n"
				   "dam = a*0;\n"
				   "for z = 1:size(dam,1)\n"
				   "  dam(z,:) = tmp(:,:,z) * (grad(z,:)');\n"
				   "end\n"
			);
		} else {
			matlab("dam = dsm;");
		}
		matlabCmp("ds", "dsm");
		matlabCmp("da", "dam");
	} else {
		matlab( "tmp = eye(size(a,2));\n"
			   "sdsm = a*0;\n"
			   "tmp = eye(size(a,2));\n"
			   "for z = 1:size(a,1)\n"
			   "  jacobiano  = tmp .* (sm(z,:) .* (1 - sm(z,:))) - (1-tmp).*(sm(z,:)'*sm(z,:));\n"
			   "  sdsm(z,:) = (jacobiano*(dsm(z,:)'))';\n"
			   "end\n"
			   "jacobiano=0;");

		matlabCmp("sds", "sdsm");
		matlab("grad = sdsm;");

		if (cf->flag & SOFTNORM) {
			matlab("tmp = reshape([0:size(dsm,2)^2-1],size(dsm,2),size(dsm,2))';\n"
				   "i = floor(tmp/size(dsm,2));\n"
				   "j = mod(tmp,size(dsm,2));\n"
				   "tmp = (i==j)-(j==reshape(imxm,[1,1,size(imxm,1)]));\n"
				   "dam = a*0;\n"
				   "for z = 1:size(dam,1)\n"
				   "  dam(z,:) = tmp(:,:,z) * (grad(z,:)');\n"
				   "end\n"
			);


		} else {
			matlab("dam = grad;");
		}
		matlabCmp("ds", "dsm");
		matlabCmp("da", "dam");
	}
	fclose(f);
	Release(entrada);
	Release(target);
	return cnn->release(&cnn);
}