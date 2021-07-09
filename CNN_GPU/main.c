#include "locale.h"

#include"src/cnn.h"
#include"src/defaultkernel.h"
#define XY 10
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
	, XY, XY, 3, CL_DEVICE_TYPE_GPU);
	printf("here\n");
	CnnAddConvLayer(cnn, 0, 1, XY, 20);
	if (cnn->error.error) {
		fprintf(stderr, "%s\n", cnn->error.context);
		fprintf(stderr, "%d: %s\n", cnn->error.error, cnn->error.msg);
		goto end;
	}
	CamadaConv conv = (CamadaConv) cnn->camadas[0];
	system("pause");
	FILE *f = fopen("../debug.txt", "w");
	double entrada[XY*XY*3];
	for(int i=0;i<300*300*3;i++){
		entrada[i]=1.0/(i+1);
	}
	CnnCall(cnn,entrada);
	fprintf(f,"entrada\n");
	printTensor(cnn->queue, conv->super.entrada, f);
	fprintf(f,"saida\n");
	printTensor(cnn->queue, conv->super.saida, f);
	fprintf(f,"filtro\n");
	printTensor(cnn->queue, conv->filtros, f);
	fclose(f);

	end:

	if (cnn) {
		if (cnn->error.error) {
			fprintf(stderr, "%s\n", cnn->error.context);
			fprintf(stderr, "%d: %s\n", cnn->error.error, cnn->error.msg);
			erro = cnn->error.error;
		}
		releaseCnn(&cnn);
	}
	return erro;
}