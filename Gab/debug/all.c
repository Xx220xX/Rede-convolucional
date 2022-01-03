//
// Created by Henrique on 03/01/2022.
//

#include "cnn/cnn_lua.h"
#include "camadas/all_camadas.h"

int main() {
	Cnn c = Cnn_new();
	c->setInput(c, 5, 5, 3);
	c->Padding(c, 1, 1, 1, 1);
	c->ConvolucaoF(c, P2D(2, 2), P3D(3, 3, 2), FRELU, Params(1e-3), RDP(-1));

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
//	Tensor A = Tensor_new(5,5,3,1,c->ecx,0,c->gpu->context,c->queue);
	CamadaConvF cf = CST_CONVOLUCAOF(c, 1);
	cf->W->setvalues(cf->W, vpesos);
	cf->B->setvalues(cf->B, vbias);

	cf->W->print(cf->W);
	printf("\n");
	cf->B->print(cf->B);
	printf("\n");
	c->predictv(c, ventrada);
	printf("Entrada");
	c->entrada->print(c->entrada);
	printf("\nantes de ativar");
	cf->z->print(cf->z);
	printf("\nsaida");
	cf->super.s->print(cf->super.s);

	return c->release(&c);
}