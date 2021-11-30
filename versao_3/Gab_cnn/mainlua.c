
#include "cnn/cnn_lua.h"
int main() {
	Cnn c = Cnn_new();
	c->setInput(c, 28, 28, 1);
	c->Convolucao(c, P2D(1, 1), P3D(2, 2, 3), Params(1e-3), RDP(0));
	c->ConvolucaoF(c, P2D(1, 1), P3D(2, 2, 3),2, FTANH, Params(1e-3), RDP(0));
	c->ConvolucaoNC(c, P2D(1, 1), P2D(2, 2), P3D(2, 2, 3), FTANH, Params(1e-3), RDP(0));
	c->Pooling(c, P2D(2, 2), P2D(4, 4), MAXPOOL);
	c->Relu(c, 1e-5f, 1.0f);
	c->PRelu(c, Params(1e-3), RDP(0));
	c->BatchNorm(c, 1e-8f, Params(1e-3), RDP(0), RDP(0));
	c->Padding(c, 1, 1, 1, 1);
	c->DropOut(c, 0.8f, time(0));
	c->FullConnect(c, 10, Params(1e-3), FTANH, RDP(0), RDP(0));
	c->SoftMax(c);
	c->jsonF(c, 1, "../a.json");

	return c->release(&c);
}