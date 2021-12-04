//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "camadas/CamadaFullConnect.h"
#include<string.h>
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define matlabf(format, ...) fprintf(f,format,##__VA_ARGS__);fprintf(f,"\n")
#define matlab(x) fprintf(f,"%s\n",x)

#define matlabCmp(x, y, T)matlab("figure;hold on;"); \
matlab("plot("x""T"(:),'DisplayName','"x"')");\
matlab("plot("y""T"(:),'DisplayName','"y"')");      \
matlab("title('"#x" vs "#y"');");\
matlab("legend();");                                  \
matlab("if gcf() == 1");\
matlabf("print('%s.pdf')",__FILENAME__);                     \
matlabf("else print('%s.pdf', '-append'); end;",__FILENAME__)


int main() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(30, 40, 8, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	Tensor target = Tensor_new(1, 4, 1, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->FullConnect(cnn, 4, Params(1e-3), FTANH, RDP(0), RDP(0));
	CamadaFullConnect cf = (CamadaFullConnect) cnn->cm[0];
	entrada->randomize(entrada,TENSOR_NORMAL,1,0);
	target->randomize(target,TENSOR_NORMAL,1,0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->predict(cnn, entrada);
	FILE *f = fopen("D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\fullconnect.m", "w");
	matlab("clc;close all;clear all");
	cf->super.s->tomatlab(cf->super.s, f, "s", NULL);
	cf->super.a->tomatlab(cf->super.a, f, "a", NULL);
	cf->w->tomatlab(cf->w, f, "w", "mshape");
	cf->b->tomatlab(cf->b, f, "b", "mshape");
	cf->z->tomatlab(cf->z, f, "z", "mshape");

	cnn->learn(cnn, target);
	target->tomatlab(target, f, "t", NULL);
	_tomatlab(cf->dz, f, "dz", "mshape");
	_tomatlab(cnn->ds, f, "ds", "mshape");
	_tomatlab(cf->dw, f, "dw", "mshape");
	_tomatlab(cf->super.da, f, "da",NULL);
	matlab("a = a';");
	matlab("b = b';");
	matlab("t = t';");
	matlab("zm = w*a + b;");
	matlab("sm = tanh(zm);");
	matlabCmp("z","zm","'");
	matlabCmp("s","sm","'");

	matlab("dsm = sm - t;");
	matlab("dfativa = @(x) 1 - tanh(x).^2;");
	matlab("dzm = dfativa(zm) .* dsm;");
	matlab("dwm = dzm * a';");
	matlab("dam = w'*dzm;");
	matlabCmp("da","dam","");
	matlabCmp("ds","dsm","");
	matlabCmp("dz","dzm","");
	matlabCmp("dw","dwm","");
	matlabf("hitLearn = %lf",(double )cf->super.params.hitlearn);
	matlabf("momento = %lf",(double )cf->super.params.momento);
	matlabf("decaimento = %lf",(double )cf->super.params.decaimento);
	matlab("wm_t = w - hitLearn * (dwm + w * decaimento);");
	matlab("bm_t = b - hitLearn * (dzm + b * decaimento);");
	cf->w->tomatlab(cf->w, f, "w_t", "mshape");
	cf->b->tomatlab(cf->b, f, "b_t", "mshape");
	matlabCmp("b_t","bm_t","");
	matlabCmp("w_t","b_t","");
	fclose(f);
	Release(entrada);
	Release(target);

	system("start D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\" );
	return cnn->release(&cnn);
}