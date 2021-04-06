//#define LOG_CNN_KERNELCALL
#include <conio.h>
#include <ctype.h>
#include "src/cnn.h"
#include "uteisTreino.h"

#define TAMANHO_IMAGEM 28
#define MAXIMO_EPOCAS_PARA_TREINAMENTO 100

#define SALVAR_REDE_A_CADA 1
#define SAIDA_REDE "../testes/redes"
#define NUMERO_DE_IMAGENS_NO_BANCO_DE_DADOS  60000
#define NUMERO_DE_IMAGENS_PARA_TREINAR 59900
#define ARQUIVO_IMAGENS "../testes/train-images.idx3-ubyte"
#define ARQUIVO_RESPOSTA "../testes/train-labels.idx1-ubyte"

int main() {
	srand(time(0));
	// criar  cnn
	Params p = {0.1, 0.0, 0.0, 1};
	Cnn c = createCnnWithgpu("../kernels/gpu_function.cl", p, TAMANHO_IMAGEM, TAMANHO_IMAGEM, 1);

	CnnAddConvLayer(c, 1, 3, 8);
	CnnAddPoolLayer(c, 1, 2);
//	CnnAddConvLayer(c, 1, 2, 6);
//	CnnAddPoolLayer(c,1,2);
	CnnAddFullConnectLayer(c, 50, FSIGMOID);
	CnnAddFullConnectLayer(c, 10, FSIGMOID);
	c->flags = CNN_FLAG_CALCULE_ERROR;


	int maximoEpocas = MAXIMO_EPOCAS_PARA_TREINAMENTO;
	int totalImagens = NUMERO_DE_IMAGENS_NO_BANCO_DE_DADOS;
	int imagensCheck = NUMERO_DE_IMAGENS_NO_BANCO_DE_DADOS - NUMERO_DE_IMAGENS_PARA_TREINAR;
	int limiteImages = NUMERO_DE_IMAGENS_PARA_TREINAR;


	int tamanhoTensorEntrada = TAMANHO_IMAGEM * TAMANHO_IMAGEM;
	int tamanhoTensorTarget = 10;

	size_t casos = 0;
	double *inputs = (double *) calloc(sizeof(double), tamanhoTensorEntrada * totalImagens),
			*targets = (double *) calloc(sizeof(double), tamanhoTensorTarget * totalImagens);
	int epoca = 0;
	double erros = 0;
	int acertos = 0;
	char buff[250] = {0};

	char b[16];
	double out[10];
	FILE *f = fopen("../treino.md", "w");
	FILE *imagens = fopen(ARQUIVO_IMAGENS, "rb");
	FILE *saidas = fopen(ARQUIVO_RESPOSTA, "rb");
	int i, r, t;
	size_t initTimeALL = clock(), initTimeLocal;
	int stop = 0;
	FILE *fredecnn;

	// ler imagens
	fread(b, 1, 16, imagens);
	fread(b, 1, 8, saidas);
	loadTargetData(targets, totalImagens, c->cl, c->queue, c->kernelInt2Vector, saidas, &casos);
	normalizeImage(inputs, totalImagens * tamanhoTensorEntrada, c->cl, c->queue, c->kerneldivInt, imagens, NULL);
	fclose(imagens);
	fclose(saidas);


	fprintf(f, "Treino rede neural, numero de threads disponiveis %d\n", c->cl->maxworks);
	fprintf(f, "numero de epocas %d, tamanho do buff %d\n", maximoEpocas, totalImagens);
	if (casos < limiteImages) {
		printf("erro ao ler imagens \n");
		goto finish;
	}

	fprintf(f, "\n|EPOCA | erro | acertos | tempo |\n|----|----|----|----|\n");
	printf("Para encerrar o treinamento pressione q\n");
	printf("Para ver informacoes do treinamento pressione i\n");
	for (; epoca < maximoEpocas; epoca++) {
		if (stop == 'q')break;

		initTimeLocal = clock();
		erros = 0;
		acertos = 0;
		fprintf(f, "|%d ", epoca);
		for (int caso = 0; caso < limiteImages; caso++) {
			if (stop == 'q')break;
			if (stop == 'i') {
				printf("\t imagem %d / %d tempo estimado para fim da epoca %fs\n\testima-se %f m para terminar o treinamento \n",
				       caso, limiteImages,
				       ((((double) (clock() - initTimeLocal)) / 1000.0) / (caso + 1.0)) * (limiteImages - caso),
				       ((((double) (clock() - initTimeALL)) / 1000.0) / (limiteImages * epoca + caso + 1.0)) *
				       ((limiteImages - caso) + limiteImages * (maximoEpocas - epoca - 1)) / 60.0);
				stop = 0;
			}
			if (kbhit())stop = tolower(getch());
			CnnCall(c, inputs + tamanhoTensorEntrada * caso);
			CnnLearn(c, targets + tamanhoTensorTarget * caso);
			for (t = 0; t < 9 && !*(targets + tamanhoTensorTarget * caso + t); t++);
			r = CnnGetIndexMax(c);
			if (r == t) { acertos++; }
			erros += c->normaErro;
		}
		printf("epoca %d, erro %g, acertos %.2lf%%\n", epoca, erros, acertos * 100.0 / limiteImages);
		fprintf(f, "| %g | %d | %llu |\n", erros, acertos, (clock() - initTimeLocal) * 1000);
		if ((epoca + 1) % SALVAR_REDE_A_CADA == 0) {
			snprintf(buff, 250,"%s/rede%d.cnn", SAIDA_REDE, (epoca + 1) / SALVAR_REDE_A_CADA);
			fredecnn = fopen(buff, "wb");
			cnnSave(c, fredecnn);
			fclose(fredecnn);
		}
	}
	printf("here\n");
	fprintf(f, "Tempo gasto para %d epocas %lf min\n", maximoEpocas, (clock() - initTimeALL) / 60.0);

	// salvar rede
	FILE *redeTreinada = fopen("../../javaDraw/redeTreinada.cnn", "wb");
	cnnSave(c, redeTreinada);
	fclose(redeTreinada);
	int globaldata[10] = {0};
	int globalAcertos[10] = {0};
	double globalerros[10] = {0};
	//avaliar rede
	int tacertos = 0;
	stop = 0;
	for (i = limiteImages; i < limiteImages + imagensCheck; i++) {
		if (stop == 'q')break;
		if (kbhit())stop = tolower(getch());
		CnnCall(c, inputs + tamanhoTensorEntrada * i);
		r = CnnGetIndexMax(c);
		for (t = 0; t < 9 && !*(targets + tamanhoTensorTarget * i + t); t++);
		globaldata[t]++;
		globalerros[t] += c->normaErro;
		snprintf(buff, 250, "../testes/imgs/im%d.ppm", i - limiteImages + 1);
		salveCnnOutAsPPM(c, buff);
		if (r == t) {
			globalAcertos[t]++;
			tacertos++;
		}
	}

	printf("acertos %d/%d %.2lf%%\n", tacertos, imagensCheck, (double) tacertos / (imagensCheck + 0.001) * 100.0);
	fprintf(f, "\n#Fitnes\n");
	fprintf(f, "acertos %d/%d %.2lf%%\n", tacertos, imagensCheck, (double) tacertos / (imagensCheck + 0.001) * 100.0);
	fprintf(f, "\n ESTATISTICAS \n");

	fprintf(f, "| :----: | :----: | :----: | :----: |\n");
	fprintf(f, "| Numero | total | acertos | erro quadratico |\n");
	for (i = 0; i < 10; i++) {
		fprintf(f, "| %d | %d | %d | %g |\n", i, globaldata[i], globalAcertos[i], globalerros[i]);
	}
	if (f != stdout) fclose(f);
	finish:
	if (inputs)free(inputs);
	if (targets)free(targets);
	releaseCnn(&c);
	return 0;
}

