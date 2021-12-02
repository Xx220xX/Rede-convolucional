//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "camadas/CamadaConvF.h"
#include "lcg/lcg.h"

void map(Tensor self, REAL *v, int i, int j, int z, int w, int k) {
	*v = (1.0f + k) / self->length;
}

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
matlabf("else print('%s.pdf', '-append'); end;",__FILENAME__);\

int main() {
	Cnn cnn = Cnn_new();
	LCG_setSeed(time(0));
	Tensor entrada = Tensor_new(30, 30, 8, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->ConvolucaoF(cnn, P2D(1, 1), P3D(4, 4, 6), FTANH, Params(1e-3), RDP(0));
	Tensor target = Tensor_new(unP3D(cnn->getSizeOut(cnn)), 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	CamadaConvF cf = (CamadaConvF) cnn->cm[0];
	entrada->randomize(entrada,TENSOR_NORMAL,1,0);
	target->randomize(target,TENSOR_NORMAL,1,0);

	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->erro, 0, cnn->gpu->context, cnn->queue);
	cnn->predict(cnn, entrada);
	FILE *f = fopen("D:\\Henrique\\Rede-convolucional\\Gab\\matlab\\convf.m", "w");
	matlab("clc;close all;clear all");
	_tomatlab(cf->super.a, f, "a", "mshape");
	_tomatlab(cf->w, f, "w", "mshape");
	_tomatlab(cf->z, f, "z", "mshape");
	_tomatlab(cf->super.s, f, "s", "mshape");
	cnn->learn(cnn, target);
	target->tomatlab(target, f, "t", "mshape");
	_tomatlab(cf->dz, f, "dz", "mshape");
	_tomatlab(cnn->ds, f, "ds", "mshape");
	_tomatlab(cf->dw, f, "dw", "mshape");
	_tomatlab(cf->super.da, f, "da", "mshape");
	matlabf("passo = [%zu, %zu];", cf->passox, cf->passoy);
	matlab("nfilters = size(w,4);\n"
		   "zm = zeros([(size(a)(1:2)-size(w)(1:2))./ passo + 1, size(w,4)]);\n"
		   "for l = 1:size(zm,3)\n"
		   "  for i=1:size(zm,1)\n"
		   "    for j=1:size(zm,2)\n"
		   "      init_ = ([i j] - 1) .* passo+1;\n"
		   "      a_ = a(init_(1):init_(1)+size(w,1)-1,init_(2):init_(2)+size(w,2)-1,:);\n"
		   "      zm(i,j,l) = sum((a_.*w(:,:,:,l))(:));\n"
		   "    end\n"
		   "  end\n"
		   "end");
	matlab("sm =  tanh(zm);");
	matlab("dsm = sm - t;");
	matlab("dzm = (1 - tanh(zm).^2) .* dsm;");

	matlab("dwm = w*0;\n"
		   "for l = 1:size(zm,3)\n"
		   "  for i=1:size(zm,1)\n"
		   "    for j=1:size(zm,2)\n"
		   "      init_ = ([i j] - 1) .* passo+1;\n"
		   "      dwm(:,:,:,l) = dwm(:,:,:,l) + a(init_(1):init_(1)+size(w,1)-1,init_(2):init_(2)+size(w,2)-1,:) * dzm(i,j,l);\n"
		   "    end\n"
		   "  end\n"
		   "end");
	matlab("dam = a*0;\n"
		   "for l = 1:size(zm,3)\n"
		   "  for i=1:size(zm,1)\n"
		   "    for j=1:size(zm,2)\n"
		   "      init_ = ([i j] - 1) .* passo+1;\n"
		   "      dam(init_(1):init_(1)+size(w,1)-1,init_(2):init_(2)+size(w,2)-1,:) =  dam(init_(1):init_(1)+size(w,1)-1,init_(2):init_(2)+size(w,2)-1,:)  + w(:,:,:,l)* dzm(i,j,l);\n"
		   "    end\n"
		   "  end\n"
		   "end");
	matlabCmp("z", "zm", "");
	matlabCmp("s", "sm", "");
	matlabCmp("ds", "dsm", "");
	matlabCmp("dz", "dzm", "");
	matlabCmp("dw", "dwm", "");
	matlabCmp("da", "dam", "");

	_tomatlab(cf->w, f, "w_t", "mshape");
	matlabf("hitLearn = %lf",(double )cf->super.params.hitlearn);
	matlabf("momento = %lf",(double )cf->super.params.momento);
	matlabf("decaimento = %lf",(double )cf->super.params.decaimento);
	matlab("wm_t = w - hitLearn * (dwm + w * decaimento);");
	matlabCmp("w_t", "wm_t", "");

	fclose(f);
	Release(entrada);
	Release(target);
	return cnn->release(&cnn);
}