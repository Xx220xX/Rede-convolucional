//
// Created by Henrique on 26-Jun-21.
//
#include"../include/cnn/cnn.h"
#include "../src/defaultkernel.h"

void printTensor(Tensor t, cl_command_queue queue, int ofset);

void testConv();

int main() {
	testConv();
	return 0;
}

void testConv() {
	LCG_setSeed(1);
	Params p = {0.1, 0, 0};
	Cnn c = createCnnWithWrapperProgram(default_kernel, p, 3, 3, 3, CL_DEVICE_TYPE_GPU);

	double input[27];
	double target[8];
	double pesos[12];
	for (int i = 0; i < 27; i++) input[i] = i / 27.0;
	for (int i = 0; i < 12; i++) pesos[i] = (i + 1) / (i + 24.0);
	for (int i = 0; i < 8; i++) target[i] = (i + 1) / 49.0;

	CnnAddConvLayer(c, 1, 2, 2);
	CamadaConv cv = (CamadaConv) c->camadas[0];

	TensorPutValuesOffSet(c->queue, cv->filtros, pesos, 0);
	for (int i = 0; i < 12; i++) pesos[i] = (i + 1) / (i + 12.0);
	TensorPutValuesOffSet(c->queue, cv->filtros, pesos, cv->filtros->bytes);
//	printCnn(c);
//	printf("Filtros \n");
//	printTensor(cv->filtros, c->queue, 0);
//	printTensor(cv->filtros, c->queue, cv->filtros->bytes);

	CnnCall(c, input);
//	printf("Saida \n");
//	printTensor(c->camadas[c->size - 1]->saida, c->queue, 0);

	CnnLearn(c,target);
//	printf("Erro \n");
//	printTensor(c->lastGrad, c->queue, 0);
//	printf("\n");
//
	printf("Grad Filtros \n");
	printTensor(cv->grad_filtros, c->queue, 0);
	printf("\n");
	printTensor(cv->grad_filtros, c->queue, cv->grad_filtros->bytes);

//	printCnn(c);
//	printf("Filtros \n");
//	printTensor(cv->filtros, c->queue, 0);
//	printTensor(cv->filtros, c->queue, cv->filtros->bytes);
	printf("Gradin \n");
	printTensor(cv->super.gradsEntrada, c->queue, 0);
	releaseCnn(&c);
}

void printTensor(Tensor t, cl_command_queue queue, int ofset) {
	double *v = (double *) calloc(t->bytes, 1);
	TensorGetValuesOffset(queue, t, ofset, v);
	for (int k = 0; k < t->z; ++k) {
		printf("dim %d\n", k);
		for (int i = 0; i < t->x; ++i) {
			for (int j = 0; j < t->y; ++j) {
				printf("%.4f ", v[TensorMap(t, i, j, k)]);
			}
			printf("\n");
		}
		printf("\n");
	}
	free(v);
}