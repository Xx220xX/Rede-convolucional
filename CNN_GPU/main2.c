//#define LOG_CNN_KERNELCALL
#include <conio.h>
#include <ctype.h>
#include "src/cnn.h"
#include "uteisTreino.h"
#include"conioHUD.h"
#include "src/lua/lua.h"
#include "src/lua/lualib.h"
#include "src/lua/lauxlib.h"
#include "src/CnnLua.h"


int loadSamples(Cnn cnn, double **images, double **labels, unsigned char **labelsI, char *imageFile, char *labelFile,
                size_t numberOfLabels, size_t numberOfSamples, size_t remainImage, size_t remainLabel) {
	int err = 0;
	size_t pixelsByImage = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	int numeroClasse = cnn->camadas[cnn->size-1]->entrada->x * cnn->camadas[cnn->size-1]->entrada->y * cnn->camadas[cnn->size-1]->entrada->z;

	size_t samples;
	*images = (double *) calloc(sizeof(double), pixelsByImage * numberOfSamples);
	*labels = (double *) calloc(sizeof(double), numberOfLabels * numberOfSamples);
	*labelsI = (unsigned char *) calloc(sizeof(unsigned char), numberOfLabels * numberOfSamples);
	FILE *fimage = fopen(imageFile, "rb");

	fread(*images, 1, remainImage, fimage);// bytes remanessentes de cabeçalho
	normalizeImage(*images, numberOfSamples
	                        * pixelsByImage, cnn->cl, cnn->queue, cnn->kerneldivInt, fimage, &samples);
	fclose(fimage);

	if (numberOfSamples * pixelsByImage != samples) {
		err = 1;
		goto error;
	}

	FILE *flabel = fopen(labelFile, "rb");
	fread(*labels, 1, remainLabel, fimage);
	loadTargetData(*labels, *labelsI, numberOfSamples,numeroClasse, cnn->cl, cnn->queue, cnn->kernelInt2Vector, flabel, &samples);
	fclose(flabel);

	if (numberOfSamples != samples) {
		err = 1;
		goto error;
	}

	error:
	if (err) {
		free(*images);
		free(*labels);
		free(*labelsI);
		*images = *labels = NULL;
		fprintf(stderr, "Error while try read data\n");
		return -2;
	}
	return 0;
}

int train(Cnn cnn, double *images, double *labels, unsigned char *labelsI, int epocs, int saveCNN, int samples,
          char *outputMDTable) {
	int caso = 0;
	int acertos = 0;
	int epoca = 0;
	int threadInfoStop = 0;
	size_t initTime = getms();
	size_t initTimeAll = getms();
	double erro = 0.0;
	InfoTrain info = {&samples, &caso, &acertos, &epoca, &epocs, &threadInfoStop, &initTime, &initTimeAll, &erro};
	pthread_t tid;
	FILE *f;
	f = fopen(outputMDTable, "w");
	if (!f)return -1;
	fprintf(f, "\n|EPOCA | erro | acertos | tempo |\n|----|----|----|----|\n");

	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->entrada->x * cnn->camadas[cnn->size - 1]->entrada->y *
	                    cnn->camadas[cnn->size - 1]->entrada->z;

	pthread_create(&tid, NULL, (void *(*)(void *)) showInfoTrain, (void *) &info);
	int r;
	for (; epoca < epocs; epoca++) {
		if (kbhit() && tolower(getch()) == 'q')break;
		initTime = getms();
		erro = 0;
		acertos = 0;
		fprintf(f, "|%d ", epoca);
		for (caso = 0; caso < samples; caso++) {
			if (kbhit() && tolower(getch()) == 'q')break;
			CnnCall(cnn, images + inputSize * caso);
			CnnLearn(cnn, labels + outputSize * caso);
			r = CnnGetIndexMax(cnn);
			if (r == labelsI[caso]) { acertos++; }
			erro += cnn->normaErro;
		}
		fprintf(f, "| %g | %d | %llu |\n", erro, acertos, (unsigned long long int) (getms() - initTime) * 1000);
	}
	fclose(f);
	threadInfoStop = 1;
	pthread_join(tid, NULL);
	return 0;
}

int fitness(Cnn cnn, double *images, unsigned char *labelsI, int nClass, Nomes *names, int samples, size_t imagesSaveOutput,
        char *path, char *outputMDTable) {
	int caso = 0;
	int acertos = 0;
	int *acertosPorClasse = calloc(sizeof(int), nClass);
	int *numeroCasosOcorridos = calloc(sizeof(int), nClass);
	double *erroPorClasse = calloc(sizeof(double), nClass);
	int threadStop = 0;
	size_t initTime = getms();
	InfoTeste info = {nClass, &samples, &caso, &acertos, &threadStop, &initTime, acertosPorClasse, numeroCasosOcorridos,
	                  erroPorClasse, names, 20};
	pthread_t tid;


	pthread_create(&tid, NULL, (void *(*)(void *)) showInfoTrain, (void *) &info);
	int r;


	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->entrada->x * cnn->camadas[cnn->size - 1]->entrada->y *
	                    cnn->camadas[cnn->size - 1]->entrada->z;

	char buff[250];
	int imPrint = 0;
	for (caso = 0; caso < samples; caso++) {
		CnnCall(cnn, images + inputSize * caso);
		r = CnnGetIndexMax(cnn);
		numeroCasosOcorridos[labelsI[caso]]++;
		erroPorClasse[labelsI[caso]] += cnn->normaErro;
		if (imPrint < imagesSaveOutput) {
			snprintf(buff, 250, "../testes/imgs/im%d.ppm", caso + 1);
			salveCnnOutAsPPM(cnn, buff);
		}
		if (r == labelsI[caso]) {
			acertosPorClasse[r]++;
			acertos++;
		}
	}
	FILE *f;
	f = fopen(outputMDTable, "w");
	if (!f)return -1;
	fprintf(f, "\n#Fitnes\n");
	fprintf(f, "acertos %d/%d %.2lf%%\n", acertos, samples, (double) acertos / (samples + 0.000001) * 100.0);
	fprintf(f, "\n ESTATISTICAS \n\n");

	fprintf(f, "| Classe | total | acertos | erro quadratico |\n");
	fprintf(f, "| :----: | :----: | :----: | :----: |\n");
	for (int i = 0; i < 10; i++) {
		fprintf(f, "| %s | %d | %d | %g |\n", names[i].names, numeroCasosOcorridos[i], acertosPorClasse[i], erroPorClasse[i]);
	}
	fclose(f);
}




Cnn loadLuaParameters(char *luaFile) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);

	luaL_loadstring(L,luaProgram)
	return NULL;
}

int main(int nargs, char **args) {

	return 0;
}

