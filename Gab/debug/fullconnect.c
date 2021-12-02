//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "camadas/CamadaFullConnect.h"

void map(Tensor self, REAL *v, int i, int j, int z, int w, int k) {
	*v = (1.0f + k) / self->length;
}

#define matlab(x) fprintf(f,"%s\n",x)
#define matlabCmp(x, y)matlab("figure;hold on;"); \
matlab("plot("x"'(:),'DisplayName','"x"')");\
matlab("plot("y"'(:),'DisplayName','"y"')");      \
matlab("title(sprintf('%f',var(("x"-"y")(:))));");\
matlab("legend();")

int main() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(30, 40, 8, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	Tensor target = Tensor_new(1, 4, 1, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->FullConnect(cnn, 4, Params(1e-3), FTANH, RDP(0), RDP(0));
	CamadaFullConnect cf = (CamadaFullConnect) cnn->cm[0];
	entrada->map(entrada, FMAP map);
	target->map(target, FMAP map);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->predict(cnn, entrada);
	FILE *f = fopen("D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\data.m", "w");
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
	matlabCmp("z","zm");
	matlabCmp("s","sm");

	matlab("dsm = sm - t;");
	matlab("dfativa = @(x) 1 - tanh(x).^2;");
	matlab("dzm = dfativa(zm) .* dsm;");
	matlab("dwm = dzm * a';");
	matlab("dam = w'*dzm;");
	matlabCmp("da","dam");
	matlabCmp("ds","dsm");
	matlabCmp("dz","dzm");
	matlabCmp("dw","dwm");

	fclose(f);
	Release(entrada);
	Release(target);
	return cnn->release(&cnn);
}