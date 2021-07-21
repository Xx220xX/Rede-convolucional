//
// Created by Henrique on 10-Jul-21.
//

#ifndef CNN_GPU_TESTECNN_H
#define CNN_GPU_TESTECNN_H
#include "locale.h"
#include"src/cnn.h"
#include"src/defaultkernel.h"
#define XY 3
#define Z 3
#define L 2
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
	cnn = createCnnWithWrapperProgram(default_kernel, (Params) {0.1, 0.3, 0}
			, XY, XY, Z, CL_DEVICE_TYPE_GPU);

	CnnAddConvLayer(cnn, 0, 1, XY, L);
	CamadaConv conv = (CamadaConv) cnn->camadas[0];

	double entrada[XY*XY*Z];
	for(int i=0;i<XY*XY*Z;i++){
		entrada[i]=1.0/(i+1);
	}
	CnnCall(cnn,entrada);
	return erro;
}
#endif //CNN_GPU_TESTECNN_H
