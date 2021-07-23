//
// Created by Henrique on 10-Jul-21.
//

#ifndef CNN_GPU_TESTECNN_H
#define CNN_GPU_TESTECNN_H

#include "locale.h"
#include"cnn/cnn.h"
#include"../src/defaultkernel.h"

#define XY 4
#define Z 3
#define L 3

int main(int nargs, char **args) {
	system("chcp 65001");
	printf("##############################\n");
	printf("Gabriela IA\n");
	printf("email: gab.cnn.ia@gmail.com\n");
	printf("Versão %s\n", getVersion());
	printf("##############################\n");
	// incializa a biblioteca random
	LCG_setSeed(1);
	// variavel auxiliar para manipular strings
	int erro;
	//  variavel cnn
	Cnn cnn = NULL;
	cnn = createCnnWithWrapperProgram(default_kernel, (Params) {0.1, 0.3, 0}, XY, XY, Z, CL_DEVICE_TYPE_GPU);

	CnnAddConvLayer(cnn, 0, 1, 2, L);

	double entrada[XY * XY * Z] = {0};
	for (int i = 0; i < XY * XY * Z; i++) entrada[i] = 1.0 / (i + 1);

	Tensor saida = newTensor(NULL, NULL, cnn->camadas[0]->saida->x, cnn->camadas[0]->saida->y,
	                         cnn->camadas[0]->saida->z, TENSOR_HOST, &cnn->error);
	double *v = saida->host;
	for (int i = saida->bytes / sizeof(double) - 1; i >= 0; i--)
		v[i] = (0.25 * (i % 4) + 0.33 * (i / 4.0) + 1) / (saida->bytes);

	double errograd, erroEntrada;
	for (int ep = 0; ep < 1000; ep++) {
		CnnCall(cnn, entrada);
		CnnLearn(cnn, v);
		CnnCalculeError(cnn);
		TensorGetNorm(cnn->queue,cnn->camadas[0]->gradsEntrada,&errograd);
		printf("%lf %lf\n",cnn->normaErro,errograd);
	}
	releaseTensor(&saida);
	releaseCnn(&cnn);
	return erro;
}

#endif //CNN_GPU_TESTECNN_H
