//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "matlab.h"
#include "camadas/all_camadas.h"

#define Tprint(t,n)printf(n);t->print(t)
int main() {

	Cnn cnn = Cnn_new();
	Tensor entrada = Tensor_new(1, 10, 1, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	cnn->DropOut(cnn, 0.3, time(0));
	P3d sizeout = cnn->getSizeOut(cnn);
	Tensor target = Tensor_new(unP3D(sizeout), 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	CamadaDropOut cf = CST_DROPOUT(cnn->cm[0]);
	entrada->randomize(entrada, TENSOR_GAUSSIAN, 1, 0);
	target->randomize(target, TENSOR_GAUSSIAN, 1, 0);
	cf->super.da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->setMode(cnn,1);
	cnn->predict(cnn, entrada);
	Tprint(cf->super.a,"Entrada");
	Tprint(cf->hitmap,"mapa");
	Tprint(cf->super.s,"Saida");
	uint8_t map[cf->hitmap->length];
	cf->hitmap->getvalues(cf->hitmap,map);
	int p = 0;
	for (int i = 0; i < cf->hitmap->length; ++i) {
		if(!map[i])p++;
	}
	printf("probabilidade %.4f\n",p/(1.0*cf->hitmap->length));
	/*	matlabInit();
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
 */
	return cnn->release(&cnn);

}