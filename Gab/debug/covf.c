//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "matlab.h"
#include "camadas/CamadaConvF.h"
#include<string.h>

int IsProcessRunning(const char *processName);

void matlab_pushvalues(CamadaConvF self, FILE *matlab_file, int layer_count) {
	char var1[16];
	fprintf(matlab_file, "global a_%d;global w_%d;global b_%d;global s_%d;global dw_%d;global dz_%d;global z_%d;global lr_%d;global mmt_%d;global dcy_%d;global z_%d;\n", layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count);

	snprintf(var1, 16, "w_%d", layer_count);
	if (self->W) {
		(self->W)->tomatlab(self->W, matlab_file, var1, "mshape");
		fprintf(matlab_file, "dw_%d = w_%d*0;\n", layer_count, layer_count);
	}
	snprintf(var1, 16, "b_%d", layer_count);
	if (self->B) {
		(self->B)->tomatlab(self->B, matlab_file, var1, NULL);
		fprintf(matlab_file, "%s = %s';\n", var1, var1);
		fprintf(matlab_file, "db_%d = b_%d*0;\n", layer_count, layer_count);

	}
	snprintf(var1, 16, "z_%d", layer_count);
	if (self->z) {
		(self->z)->tomatlab(self->z, matlab_file, var1, NULL);
		fprintf(matlab_file, "%s = %s';\n", var1, var1);
		fprintf(matlab_file, "dz_%d = z_%d*0;\n", layer_count, layer_count);

	}

	fprintf(matlab_file, "lr_%d = %.10f;\n", layer_count, self->super.params.hitlearn);
	fprintf(matlab_file, "mmt_%d = %.10f;\n", layer_count, self->super.params.momento);
	fprintf(matlab_file, "dcy%d = %.10f;\n", layer_count, self->super.params.decaimento);
	fprintf(matlab_file, "a_%d = [];\n", layer_count);

}

/*
void matlab_predict(CamadaConvF self, FILE *matlab_file, const char *var_input, int layer_count) {
	char var1[16];
	char var2[16];
	char var3[16];
	char var_out[16];
	fprintf(matlab_file,"a_%d = %s;\n",layer_count,var_input);
	fprintf(matlab_file,
			"%% padding\n"
			"a = [zeros(%zu,size(a,2),size(a,3));a];\n"
			"a = [a;zeros(%zu,size(a,2),size(a,3))];\n"
			"a = [zeros(size(a,1),%zu,size(a,3)),a];\n"
			"a = [a, zeros(size(a,1),%zu,size(a,3))];\n"
			"\n"
			"zm = s*0;\n"
			"px = %zu;\n"
			"py = %zu;\n"
			"for k = 0:(size(s,1)*size(s,2)-1)\n"
			"    y = mod(k,size(s,2))+1;\n"
			"    x = floor(k/size(s,2))+1;\n"
			"    for z = 1:size(s,3)\n"
			"      filtro = w(:,:,:,z);\n"
			"      ai = a((x-1)*px+1:(x-1)*px+size(filtro,1),(y-1)*py+1:(y-1)*py+size(filtro,2),:);\n"
			"      zm(x,y,z) = sum((filtro .* ai)(:));\n"
			"    end\n"
			"end\n"
			"zm =zm +b;\n", cf->pad_top, cf->pad_bottom, cf->pad_left, cf->pad_right, cf->passox, cf->passoy)

	switch (self->fa.id) {
		case FSIGMOID:
			fprintf(matlab_file, "%s = 1.0 ./ (1.0 + exp(-%s));", var_out, var3);
			break;
		case FTANH:
			fprintf(matlab_file, "%s = tanh(%s);", var_out, var3);
			break;
		case FLRELU:
			fprintf(matlab_file, "%s = (%s>=0).*%s*%.16f +(%s<0).*%s*%.16f;", var_out, var3, var3, self->fa.greater, var3, var3, self->fa.less);
			break;
		case FLIN:
			fprintf(matlab_file, "%s = %s;", var_out, var3);
			break;
		case FALAN:
			fprintf(matlab_file, "%s =  alan(%s);", var_out, var3);
			break;
		case FSOFTMAX:
			fprintf(matlab_file, "tmp = %s - max(%s(:));\ntmp = exp(tmp);", var3, var3);
			fprintf(matlab_file, "%s = tmp/sum(tmp(:));", var_out);
			fprintf(matlab_file, "%s(%s>%.10f) = %.10f;\n", var_out, var_out, 1 - self->fa.epsilon, 1 - self->fa.epsilon);
			fprintf(matlab_file, "%s(%s<%.10f) = %.10f;\n", var_out, var_out, self->fa.epsilon, self->fa.epsilon);
			break;
	}
	fprintf(matlab_file, "\n");
}


void matlab_backprop(CamadaConvF self, FILE *matlab_file, const char *var_ds, const char *var_da, int layer_count) {
	fprintf(matlab_file, "global a_%d;global w_%d;global b_%d;global s_%d;global dw_%d;global dz_%d;global z_%d;global lr_%d;global mmt_%d;global dcy_%d;global z_%d;\n", layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count);
	// calcular o dz
	switch (self->fa.id) {
		case FSIGMOID:
			fprintf(matlab_file, "dz_%d = %s .* (s_%d .*(1 - s_%d));", layer_count, var_ds, layer_count, layer_count);
			break;
		case FTANH:
			fprintf(matlab_file, "dz_%d = %s .* (1 - s_%d .^ 2);", layer_count, var_ds, layer_count);
			break;
		case FLRELU:
			fprintf(matlab_file, "dz_%d = %s .* ((z_%d>=0).*%.16f +(z_%d<0).*%.16f);",
					layer_count, var_ds, layer_count, self->fa.greater, layer_count, self->fa.less);
			break;
		case FLIN:
			fprintf(matlab_file, "dz_%d =%s;", layer_count, var_ds);
			break;
		case FALAN:
			fprintf(matlab_file, "dz_%d = %s .*dfalan(z_%d);", layer_count, var_ds, layer_count);
			break;
		case FSOFTMAX:
			fprintf(matlab_file, "dz_%d = %s;", layer_count, var_ds);
			break;
	}
	fprintf(matlab_file, "\n");
	fprintf(matlab_file, "%s = w_%d' * dz_%d;\n", var_da, layer_count, layer_count);
	fprintf(matlab_file, "dw_%d = dz_%d * (a_%d');\n", layer_count, layer_count, layer_count);
	fprintf(matlab_file, "db_%d = dz_%d;\n", layer_count, layer_count);

	fprintf(matlab_file, "b_%d = b_%d - lr_%d*db_%d;\n", layer_count, layer_count, layer_count, layer_count);
	fprintf(matlab_file, "w_%d = w_%d - lr_%d*dw_%d;\n", layer_count, layer_count, layer_count, layer_count);


}
*/
int main2() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(5, 5, 1, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->ConvolucaoF(cnn, P2D(1, 1), P3D(3, 3, 1), FATIVACAO(FLIN, 0.3, 1), 1, 1, 1, 1, Params(1e-3), RDP(3, 3, -1));
	P3d siz = cnn->getSizeOut(cnn);
	Tensor target = Tensor_new(siz.x, siz.y, siz.z, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);

	CamadaConvF cf = (CamadaConvF) cnn->cm[0];
	{
		REAL ventrada[] = {0, 1, 2, 0, 2, 2, 1, 2, 1, 1, 0, 0, 2, 1, 2, 0, 1, 0, 1, 2, 1, 0, 1, 2, 2,

						   2, 0, 2, 1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 1, 2, 0, 1, 1, 1, 1, 0,

						   1, 0, 2, 1, 1, 2, 2, 1, 0, 2, 0, 1, 0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 1, 1, 0};

		REAL vpesos[] = {0, -1, -1, 1, 1, 1, 0, 0, 1,
						 1, 0, 1, -1, 1, 0, 1, -1, 1,
						 -1, 0, 0, -1, -1, 0, 1, -1, 0,
						 1, 0, 1, 0, 1, -1, 0, 0, 1,
						 1, 1, -1, -1, -1, -1, -1, -1, 1,
						 1, 0, 0, 1, 1, -1, -1, -1, -1};

		REAL vbias[] = {1, 0};
//		cf->W->setvalues(cf->W, vpesos);
//		cf->B->setvalues(cf->B, vbias);
		entrada->setvalues(entrada, ventrada);

	}
	target->randomize(target, TENSOR_UNIFORM, 1, 0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);


	matlabInit();
#define N 1000

	cnn->predict(cnn, entrada);
	Trmatlab(cf->W, "w");
	Trmatlab(cf->B, "b");
	Trmatlab(entrada, "a");
	Trmatlab(target, "t");
	Trmatlab(cf->z, "Z");
	Trmatlab(cf->super.s, "s");
	cnn->learnBatch(cnn, target,1);
	Trmatlab(cf->super.da, "da");
	Trmatlab(cnn->ds, "ds");
	Trmatlab(cf->dz, "dZ");
	Trmatlab(cf->dW, "dw");
	Trmatlab(cf->dB, "dB");
	cnn->learn(cnn, target);

//	cnn->fixBatch(cnn);

	Trmatlab(cf->W, "w1");


	matlabf("%% padding\n"
			"a = [zeros(%zu,size(a,2),size(a,3));a];\n"
			"a = [a;zeros(%zu,size(a,2),size(a,3))];\n"
			"a = [zeros(size(a,1),%zu,size(a,3)),a];\n"
			"a = [a, zeros(size(a,1),%zu,size(a,3))];\n"
			"\n"
			"zm = s*0;\n"
			"px = %zu;\n"
			"py = %zu;\n", cf->pad_top, cf->pad_bottom, cf->pad_left, cf->pad_right, cf->passox, cf->passoy)

	matlabf("tst = [];\n"
			"for iter =0:%d\n", N - 1);
	matlabf("for k = 0:(size(s,1)*size(s,2)-1)\n"
			"    y = mod(k,size(s,2))+1;\n"
			"    x = floor(k/size(s,2))+1;\n"
			"    for z = 1:size(s,3)\n"
			"      filtro = w(:,:,:,z);\n"
			"      ai = a((x-1)*px+1:(x-1)*px+size(filtro,1),(y-1)*py+1:(y-1)*py+size(filtro,2),:);\n"
			"      zm(x,y,z) = sum((filtro .* ai)(:));\n"
			"    end\n"
			"end\n"
			"zm =zm +b;\n")
	matlabAtivation("sm", "zm", cf->fa.id);

//	matlabCmp("Z", "zm");
//	matlabCmp("s", "sm");


	matlab("dsm = sm - t;");
	matlabAtivation("dzm", "zm", cf->fa.id + 1);
	matlab("dzm = dsm .*dzm;");

	matlab("dam = a*0;\n"
		   "for k = 0:(size(s,1)*size(s,2)-1)\n"
		   "    y = mod(k,size(s,2))+1;\n"
		   "    x = floor(k/size(s,2))+1;\n"
		   "    for z = 1:size(s,3)\n"
		   "      filtro = w(:,:,:,z);\n"
		   "      dam((x-1)*px+1:(x-1)*px+size(filtro,1),(y-1)*py+1:(y-1)*py+size(filtro,2),:) =  dam((x-1)*px+1:(x-1)*px+size(filtro,1),(y-1)*py+1:(y-1)*py+size(filtro,2),:)+ filtro .* dzm(x,y,z);      \n"
		   "    end\n"
		   "end");
	matlabf("dam = dam(1+%zu:%zu,1+%zu:%zu,:);\n", cf->pad_top, cf->pad_top + entrada->x, cf->pad_left, cf->pad_left + entrada->y);
	matlab("dwm = w*0;\n"
		   "for k = 0:(size(s,1)*size(s,2)-1)\n"
		   "    y = mod(k,size(s,2))+1;\n"
		   "    x = floor(k/size(s,2))+1;\n"
		   "    for z = 1:size(s,3)\n"
		   "      ai = a((x-1)*px+1:(x-1)*px+size(filtro,1),(y-1)*py+1:(y-1)*py+size(filtro,2),:);\n"
		   "      dwm(:,:,:,z) = dwm(:,:,:,z) +   ai .* dzm(x,y,z);\n"
		   "    end\n"
		   "end\n");
	matlab("dbm = b*0;");
	matlab("for z=1:size(dzm,3)\n"
		   "dbm(z) = sum(dzm(:,:,z)(:));\n"
		   "end");
	matlabf("w = w - %.10lf*dwm;\n", cf->super.params.hitlearn);
	matlabf("b = b - %.10lf*dbm;\n", cf->super.params.hitlearn);
	matlab("tst = [tst var(dsm(:),1)];\nend");
//	matlabCmp("w", "w1");
//	matlabCmp("dw", "dwm");
//	matlabCmp("da","dam");
//	matlabCmp("dZ","dzm");
//	matlabCmp("ds","dsm");


	matlabf("dt = [0");
	for (int i = 0; i < N; i++) {
		cnn->predict(cnn, entrada);
		cnn->learnBatch(cnn, target,1);
		cnn->fixBatch(cnn);
		matlabf(", %.10f", cnn->ds->var(cnn->ds));
	}
	matlabf("];\n");
	matlab("figure");
	matlab("dt = dt(2:end);");
	matlab("plot(dt,'DisplayName','C');hold on");
	matlab("plot(tst,'DisplayName','matlab');hold on");
	matlab("legend();");

	matlabEnd();
	if (!IsProcessRunning("octave-gui")) {
		system("start ../matlab/covf.m");
//		system("start debug.m");
	}

	Release(target);
	Release(entrada);
	return cnn->release(&cnn);
}

int main() {
	LCG_setSeed(time(0));
	Cnn cnn = Cnn_new();
	Tensor target;
	Tensor entrada = Tensor_new(5, 5, 1, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	int i = 0;
	FILE *f = fopen("convtest.m", "w");
	fprintf(f, "clc;close all;clear all;\nhold on;\n");
	struct {
		char *name;
		FAtivacao f;
	} funcoes[] = {{"linear",     FATIVACAO(FLIN)},
				   {"tanh",       FATIVACAO(FTANH)},
				   {"relu",       FATIVACAO(FRELU)},
				   {"leaky_relu", FATIVACAO(FLRELU, 0.1, 1)},
				   {"sigmoid",    FATIVACAO(FSIGMOID)},
				   {"alan",       FATIVACAO(FALAN)},
				   {NULL}
	};
	putLayer:
//	cnn->ConvolucaoF(cnn, P2D(1, 1), P3D(3, 3, 1), funcoes[i].f.mask, 1, 1, 1, 1, Params(1e-3), RDP(3, 3, -1));
	cnn->FullConnect(cnn, 5,  Params(1e-3),funcoes[i].f.mask, RDP(0, 3, -1), RDP(0, 3, -1));
	if (i != 0) {
		goto avaliar;
	}
	int batchSize = 20;

	P3d siz = cnn->getSizeOut(cnn);
	target = Tensor_new(siz.x, siz.y, siz.z, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	entrada->randomize(entrada, TENSOR_RANDINT, 3, -1);
	target->randomize(target, TENSOR_GAUSSIAN | TENSOR_UNITARIO, 0, 0);
	avaliar:
	cnn->predict(cnn, entrada);
	cnn->learn(cnn, target);
	fprintf(f, "ativacao_%s = [%.10lf", funcoes[i].name, cnn->ds->var(cnn->ds));
	int k = 0;
	for (int j = 0; j < 3000; ++j) {
		cnn->predict(cnn, entrada);
//		cnn->learn(cnn,target);
		k++;
		cnn->learnBatch(cnn, target, batchSize);
		if (k > batchSize) {
			cnn->fixBatch(cnn);
			k = 0;
		}
		fprintf(f, ", %.10lf", cnn->ds->var(cnn->ds));
	}
	fprintf(f, "];\nplot(ativacao_%s, 'DisplayName','%s');\n", funcoes[i].name, funcoes[i].name);
	fprintf(f, "printf('minimo %s = %%f\\n',min(ativacao_%s));\n", funcoes[i].name, funcoes[i].name);
	i++;
	if (funcoes[i].name) {
		cnn->removeLastLayer(cnn);
		goto putLayer;
	}
	fprintf(f, "legend()\n");
	fclose(f);
	Release(target);
	Release(entrada);
	return cnn->release(&cnn);
}
