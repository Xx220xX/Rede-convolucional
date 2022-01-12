//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "matlab.h"
#include "camadas/CamadaFullConnect.h"
#include<string.h>

int IsProcessRunning(const char *processName);

void matlab_pushvalues(CamadaFullConnect self, FILE *matlab_file, int layer_count) {
	char var1[16];
	fprintf(matlab_file, "global a_%d;global w_%d;global b_%d;global s_%d;global dw_%d;global dz_%d;global z_%d;global lr_%d;global mmt_%d;global dcy_%d;global z_%d;\n", layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count);

	snprintf(var1, 16, "w_%d", layer_count);
	if (self->w) {
		(self->w)->tomatlab(self->w, matlab_file, var1, "mshape");
		fprintf(matlab_file, "dw_%d = w_%d*0;\n", layer_count, layer_count);
	}
	snprintf(var1, 16, "b_%d", layer_count);
	if (self->b) {
		(self->b)->tomatlab(self->b, matlab_file, var1, NULL);
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

void matlab_predict(CamadaFullConnect self, FILE *matlab_file, const char *var_input, int layer_count) {
	char var1[16];
	char var2[16];
	char var3[16];
	char var_out[16];
	snprintf(var1, 16, "w_%d", layer_count);
	snprintf(var2, 16, "b_%d", layer_count);
	snprintf(var3, 16, "z_%d", layer_count);
	snprintf(var_out, 16, "s_%d", layer_count);
	fprintf(matlab_file, "global a_%d;global w_%d;global b_%d;global s_%d;global dw_%d;global dz_%d;global z_%d;global lr_%d;global mmt_%d;global dcy_%d;global z_%d;\n", layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count, layer_count);

	fprintf(matlab_file, "%s = %s * %s + %s;\n", var3, var1, var_input, var2);
	fprintf(matlab_file, "a_%d = %s;\n", layer_count, var_input);
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
			fprintf(matlab_file, "%s =  alan(%s);",var_out,var3);
			break;
		case FSOFTMAX:
			fprintf(matlab_file, "tmp = %s - max(%s(:));\ntmp = exp(tmp);", var3, var3);
			fprintf(matlab_file, "%s = tmp/sum(tmp(:));", var_out);
			fprintf(matlab_file, "%s(%s>%.10f) = %.10f;\n", var_out, var_out,1-self->fa.epsilon,1-self->fa.epsilon);
			fprintf(matlab_file, "%s(%s<%.10f) = %.10f;\n", var_out, var_out,self->fa.epsilon,self->fa.epsilon);
			break;
	}
	fprintf(matlab_file, "\n");
}

void matlab_backprop(CamadaFullConnect self, FILE *matlab_file, const char *var_ds, const char *var_da, int layer_count) {
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

int main() {
	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(2, 2, 3, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	Tensor target = Tensor_new(1, 4, 1, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->FullConnect(cnn, 4, Params(1e-2), FATIVACAO(FLIN, 0.1, 1), RDP(0), RDP(0));
	CamadaFullConnect cf = (CamadaFullConnect) cnn->cm[0];
	entrada->randomize(entrada, TENSOR_GAUSSIAN, 1, 0);
	target->randomize(target, TENSOR_UNIFORM, 1, 0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->predict(cnn, entrada);


	matlabInit();
	matlab("clc;close all;clear all");
	Tmatlab(cf->super.s, "s");
	Tmatlab(cf->super.a, "a");
	Tmatlab(target, "t");
	matlab("a = a';t = t';");
	matlab_pushvalues(cf, _matlab_file_, 0);
	matlab("function predict(a)");
	matlab_predict(cf, _matlab_file_, "a", 0);
	matlab("end");
	matlab("function back(ds)");
	matlab_backprop(cf, _matlab_file_, "ds", "da", 0);
	matlab("end");

	matlab("gp = [];\n"
		   "for i = 1:100\n"
		   "  predict(a);\n"
		   "  ds = s_0 - t;\n"
		   "  gp = [gp var(ds(:),1)];\n"
		   "  plot(gp)\n"
//		   "  drawnow;\n"
		   "  back(ds);\n"
		   "end");
	matlabf("cn =[0")
	for (int i = 0; i < 100; ++i) {
		cnn->predict(cnn, entrada);
		cnn->learn(cnn, target);
		matlabf(", %.10f",cnn->ds->var(cnn->ds));
	}
	matlabf("];\ncn = cn(2:end);\n");
	matlab("figure;");
	matlab("plot(gp);hold on;");
	matlab("plot(cn);");
	matlab("legend('matlab','C')");

	matlabEnd();
	Release(entrada);
	Release(target);
	if (!IsProcessRunning("octave-gui")) {
		system("start ../matlab/fullconnect.m");
	}
	return cnn->release(&cnn);
}


