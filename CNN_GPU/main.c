#include "utils/manageTrain.h"
#include "src/defaultkernel.h"
#include "utils/vectorUtils.h"
// call backs


void printALLIMGS(ManageTrain *t) {
	fflush(stderr);
	fflush(stdout);
	char name[100];
	int len;
	double *v;
	char *aux;
	for (int w = 0; w < t->imagens->w && w<300; w++) {
		len = snprintf(name, 100, "l%d_%d[",w, (int) ((char *) t->labels->host)[0]);

		v = t->targets->host+ w*t->targets->y*sizeof(double);
		aux = name;
		for (int i = 0; i < t->n_classes; i++) {
			aux = aux + len;
			len = snprintf(aux, 100 - (size_t) aux + (size_t) name, "%d_", (int) v[i]);
		}
		aux = aux + len;
		len = snprintf(aux, 100 - (size_t) aux + (size_t) name, "].ppm");

//		salveTensor4DAsPPM3(name, t->imagens, t->cnn, w);
	}
}


#define PATH "C:\\Users\\Henrique\\Desktop\\last\\TESTES_REDE_CONVOLUCIONAL\\treino_10classes\\"

int main(int arg, char **args) {
	ManageTrain manageTrain = {0};

	manageTrain.cnn = createCnnWithWrapperProgram(default_kernel, (Params) {0.1, 0, 0},
	                                              32, 32, 3, CL_DEVICE_TYPE_GPU);
	manageTrain.OnloadedImages = (ManageEvent) printALLIMGS;

	manageTrain.n_classes = 10;
	manageTrain.n_images = 300;

	manageTrain.file_images = PATH "imagesCifar10.ubyte";
	manageTrain.file_labels = PATH "labelsCifar10.ubyte";

	Convolucao(manageTrain.cnn, 0, 2, 2, 2, 2, 2);
	loadImages(&manageTrain);
	if (manageTrain.cnn->error.error) {
		printf("%s\n", manageTrain.cnn->error.msg);
	}
	releaseManageTrain(&manageTrain);
	return 0;
}