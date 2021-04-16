//#define LOG_CNN_KERNELCALL
#include <conio.h>
#include <ctype.h>
#include <windows.h>
#include "src/cnn.h"
#include "uteisTreino.h"
#include"conioHUD.h"
#include "src/lua/lua.h"
#include "src/lua/lualib.h"
#include "src/lua/lauxlib.h"
#include "src/CnnLua.h"

int loadImage(Cnn cnn, double **images, size_t remainImage, size_t numberOfSamples, char *imageFile) {
	size_t pixelsByImage = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t samples;

	FILE *fimage = fopen(imageFile, "rb");
	if (!fimage) {
		fprintf(stderr, "Imagens nao foram encontradas em %s\n", imageFile);
		*images = NULL;
		return -1;
	}
	*images = (double *) calloc(sizeof(double), pixelsByImage * numberOfSamples);
	fread(*images, 1, remainImage, fimage);// bytes remanessentes de cabeçalho
	normalizeImage(*images, numberOfSamples * pixelsByImage,
	               cnn->cl, cnn->queue, cnn->kerneldivInt, fimage, &samples);
	fclose(fimage);

	if (numberOfSamples * pixelsByImage != samples) {
		fprintf(stderr, "As imagens nao foram lidas corretamente\n");
		free(*images);
		*images = NULL;
		return -2;
	}
	return 0;

}

int loadLabel(Cnn cnn, double **labels, unsigned char **labelsI, size_t remainLabel, size_t numberOfSamples,
              size_t numeroSaidas, char *labelFile) {
	FILE *flabel = fopen(labelFile, "rb");
	if (!flabel) {
		fprintf(stderr, "Labels nao foram encontradas em %s\n", labelFile);
		*labels = NULL;
		*labelsI = NULL;
		return -1;
	}
	*labels = (double *) calloc(sizeof(double), numeroSaidas * numberOfSamples);
	*labelsI = (unsigned char *) calloc(sizeof(unsigned char), numberOfSamples);

	fread(*labels, 1, remainLabel, flabel);
	size_t lidos = 0;
	loadTargetData(*labels, *labelsI, numeroSaidas, numberOfSamples, cnn->cl, cnn->queue, cnn->kernelInt2Vector, flabel,
	               &lidos);
	fclose(flabel);

	if (numberOfSamples != lidos) {
		fprintf(stderr, "Esperado %lld, lidos %lld\n", numberOfSamples, lidos);
		*labels = NULL;
		*labelsI = NULL;
		return -2;
	}
	return 0;
}

int loadSamples(Cnn cnn, double **images, double **labels, unsigned char **labelsI, char *imageFile, char *labelFile,
                size_t numberOfLabels, size_t numberOfSamples, size_t remainImage, size_t remainLabel) {


	if (loadImage(cnn, images, remainImage, numberOfSamples, imageFile))return -7;

	if (loadLabel(cnn, labels, labelsI, remainLabel, numberOfSamples, numberOfLabels, labelFile))return -8;
	return 0;
}

void printTestImages(double *images, double *labels, unsigned char *labelsI, int n, int x, int y, int z, int m) {
	char buff[500];
	system("mkdir \"imgTeste\"");
	for (int i = 0; i < n; i++) {
		snprintf(buff, 500, "imgTeste/[%d][%d] ", (int) labelsI[i], m);
		for (int j = 0; j < m; j++) {
			snprintf(buff, 500, "%s%.1lf,", buff, labels[i * m + j]);
		}
		snprintf(buff, 500, "%s.ppm", buff);
		ppmp2(images + i * x * y * z, x, y, buff);
	}
}

int train(Cnn cnn, double *images, double *labels, unsigned char *labelsI, int epocs, int saveCNN, int samples,
          char *outputMDTable) {
	int caso = 0;
	int acertos = 0;
//	printTestImages(images, labels, labelsI, samples, cnn->camadas[0]->entrada->x, cnn->camadas[0]->entrada->y,
//	                cnn->camadas[0]->entrada->z, cnn->camadas[cnn->size - 1]->saida->x);
//	system("pause");
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
	size_t outputSize = cnn->camadas[cnn->size - 1]->saida->x * cnn->camadas[cnn->size - 1]->saida->y *
	                    cnn->camadas[cnn->size - 1]->saida->z;

	pthread_create(&tid, NULL, (void *(*)(void *)) showInfoTrain, (void *) &info);
	int r;
	int stop = 0;
	for (; epoca < epocs; epoca++) {
		if (stop == 'q')break;
		initTime = getms();
		erro = 0;
		acertos = 0;
		fprintf(f, "|%d ", epoca);
		for (caso = 0; caso < samples; caso++) {
			if (kbhit() && (stop = tolower(getch())) == 'q')break;
			CnnCall(cnn, images + inputSize * caso);
			CnnLearn(cnn, labels + outputSize * caso);
			r = CnnGetIndexMax(cnn);
			if (r == labelsI[caso]) { acertos += 1; }
			erro += cnn->normaErro;
		}
		fprintf(f, "| %g | %d | %llu |\n", erro, acertos, (unsigned long long int) (getms() - initTime) * 1000);
	}
	fclose(f);
	threadInfoStop = 1;

	pthread_join(tid, NULL);
	return 0;
}

int
fitness(Cnn cnn, double *images, unsigned char *labelsI, int nClass, Nomes *names, int samples, size_t imagesSaveOutput,
        char *name, char *outputMDTable) {
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


	pthread_create(&tid, NULL, (void *(*)(void *)) showInfoTest, (void *) &info);
	int r;


	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->saida->x * cnn->camadas[cnn->size - 1]->saida->y *
	                    cnn->camadas[cnn->size - 1]->saida->z;

	char buff[250];
	int imPrint = 0;
	for (caso = 0; caso < samples; caso++) {
		if (kbhit() && tolower(getch()) == 'q')break;
		CnnCall(cnn, images + inputSize * caso);
		r = CnnGetIndexMax(cnn);
		numeroCasosOcorridos[labelsI[caso]]++;
		erroPorClasse[labelsI[caso]] += cnn->normaErro;
		if (imPrint < imagesSaveOutput) {
			snprintf(buff, 250, "imgs/%s%d.ppm", name, caso + 1);
			salveCnnOutAsPPM(cnn, buff);
			imPrint++;
		}
		if (r == labelsI[caso]) {
			acertosPorClasse[r]++;
			acertos++;
		}
	}
	threadStop = 1;
	pthread_join(tid, NULL);
	FILE *f;
	f = fopen(outputMDTable, "w");

	if (!f)return -1;
	fprintf(f, "\n#Fitnes\n");
	fprintf(f, "acertos %d/%d %.2lf%%\n", acertos, samples, (double) acertos / (samples + 0.000001) * 100.0);
	fprintf(f, "\n ESTATISTICAS \n\n");

	fprintf(f, "| Classe | total | acertos | erro quadratico |\n");
	fprintf(f, "| :----: | :----: | :----: | :----: |\n");
	for (int i = 0; i < 10; i++) {
		fprintf(f, "| %s | %d | %d | %g |\n", names[i].names, numeroCasosOcorridos[i], acertosPorClasse[i],
		        erroPorClasse[i]);
	}
	fclose(f);

}

#include<dirent.h>

void showDir() {
	DIR *dir;
	struct dirent *ent;
	char currentDir[250] = {0};
	GetCurrentDirectory(250, currentDir);
	printf("Current dir:%s\n", currentDir);
	if ((dir = opendir(currentDir)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	getchar();
}

int loadLuaParameters(char *luaFile, ParametrosCnnALL *p) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	loadCnnLuaLibrary(L);
	printf("%s\n", luaFile);
	printf("carregando script: %s\n", luaFile);
	//showDir();
	luaL_loadfile(L, luaFile);
	int error = lua_pcall(L, 0, 0, 0);

	// o scrip foi executado
	if (error) {
		fprintf(stderr, "Falha ao carregar scrip\n");
		fprintf(stderr, "stack:%d\n", lua_gettop(L));
		fprintf(stderr, "erro:%d\n", error);
		fprintf(stderr, "message:%s\n", lua_tostring(L, -1));
		system("pause");
		return error;
	}
	// verificar se as variaveis foram setadas
	char *tmp;
	GETLUAVALUE(p->Numero_epocas, L, "Numero_epocas", integer,
	            printf("ERRO: Numero_epocas nao foi atribuido");return -1;);
	GETLUAVALUE(p->Numero_Imagens, L, "Numero_Imagens", integer,
	            printf("ERRO: Numero_Imagens nao foi atribuido");return -1;);
	GETLUAVALUE(p->Numero_ImagensAvaliacao, L, "Numero_ImagensAvaliacao", integer,
	            printf("ERRO: Numero_ImagensAvaliacao nao foi atribuido");return -1;);
	GETLUAVALUE(p->Numero_ImagensTreino, L, "Numero_ImagensTreino", integer,
	            printf("ERRO: Numero_ImagensTreino nao foi atribuido");return -1;);
	GETLUAVALUE(p->SalvarSaidasComoPPM, L, "SalvarSaidasComoPPM", integer,
	            printf("ERRO: SalvarSaidasComoPPM nao foi atribuido");return -1;);
	GETLUAVALUE(p->SalvarBackupACada, L, "SalvarSaidasComoPPM", integer,
	            printf("ERRO: SalvarSaidasComoPPM nao foi atribuido");return -1;);
	GETLUAVALUE(p->Numero_Classes, L, "Numero_Classes", integer,
	            printf("ERRO: Numero_Classes nao foi atribuido");return -1;);
	GETLUAVALUE(p->bytes_remanessentes_classes, L, "bytes_remanessentes_classes", integer,
	            printf("ERRO: bytes_remanessentes_classes nao foi atribuido");return -1;);
	GETLUAVALUE(p->bytes_remanessentes_imagem, L, "bytes_remanessentes_imagem", integer,
	            printf("ERRO: bytes_remanessentes_imagem nao foi atribuido");return -1;);

	GETLUASTRING(p->nome, tmp, MAX_STRING_LEN, L, "nome", printf("ERRO: nome nao foi atribuido");return -1;);
	GETLUASTRING(p->home, tmp, MAX_STRING_LEN, L, "home", printf("ERRO: home nao foi atribuido");return -1;);

	GETLUASTRING(p->estatisticasDeTreino, tmp, MAX_STRING_LEN, L, "estatisticasDeTreino",
	             printf("ERRO: estatisticasDeTreino nao foi atribuido");return -1;);
	GETLUASTRING(p->estatiscasDeAvaliacao, tmp, MAX_STRING_LEN, L, "estatiscasDeAvaliacao",
	             printf("ERRO: estatiscasDeAvaliacao nao foi atribuido");return -1;);
	GETLUASTRING(p->arquivoContendoImagens, tmp, MAX_STRING_LEN, L, "arquivoContendoImagens",
	             printf("ERRO: arquivoContendoImagens nao foi atribuido");return -1;);
	GETLUASTRING(p->arquivoContendoRespostas, tmp, MAX_STRING_LEN, L, "arquivoContendoRespostas",
	             printf("ERRO: arquivoContendoRespostas nao foi atribuido");return -1;);

	p->names = (Nomes *) calloc(sizeof(Nomes), p->Numero_Classes);

	lua_getglobal(L, "classes"); // classes
	for (int i = 1; i <= p->Numero_Classes; i++) {
		lua_pushinteger(L, i);//classes, i
		lua_gettable(L, -2); // classes, i, classes[i]
		tmp = (char *) lua_tostring(L, -1);
		snprintf(p->names[i - 1].names, MAX_STRING_LEN, "%s", tmp);
		lua_pop(L, 1);
	}
	lua_close(L);
	printf("script carregado\n");
	return 0;
}

void getpath(char *fileabsolutpath, char *dst, int len_dst) {
	int len = strlen(fileabsolutpath);
	for (len = len - 1; len >= 0 && fileabsolutpath[len] != '/' && fileabsolutpath[len] != '\\'; len--);
	if (len >= len_dst || dst == NULL) {
		fprintf(stderr, "Tamanho do destino invalido");
		system("pause");
		return;
	}
	memcpy(dst, fileabsolutpath, len);
	dst[len + 1] = 0;

}

int main(int nargs, char **args) {
	srand(time(NULL));
	if (nargs != 2) {
		fprintf(stderr, "Um script lua de configuração é esperado");
		return -1;
	}
	char buff[300] = {0};
	int erro;
	getpath(args[0], buff, 300);
	if (!SetCurrentDirectory(buff)) {
		fprintf(stderr, "Falha ao mudar para o diretorio %s\n", buff);
		erro = 2;
		goto end;
	}
	Cnn cnn = NULL;
	globalcnn = &cnn;
	char kernelFile[300] = "../kernels/gpu_function.cl";
	globalKernel = kernelFile;
	char *luaFile = args[1];
	ParametrosCnnALL p = {0};
	erro = loadLuaParameters(luaFile, &p);
	double *input = NULL, *target = NULL;
	unsigned char *targeti = NULL;
	if (erro)goto end;
	if (!cnn) {
		fprintf(stderr, "Nao foi encontrado uma arquitetura de rede");
		goto end;
	}
	printf("ARQUITETURA DA REDE\n");
	globalcnn = NULL;
	printf("entrada   (%d,%d,%d)\n", cnn->camadas[0]->entrada->x, cnn->camadas[0]->entrada->y,
	       cnn->camadas[0]->entrada->z);
	printf("camada    Tipo       saida\n");
	for (int i = 0; i < cnn->size; i++) {
		printf("% 2d         ", i);
		switch (cnn->camadas[i]->type) {
			case CONV:
				printf("CONV       ");
				break;
			case RELU:
				printf("RELU       ");
				break;
			case FULLCONNECT:
				printf("FULLCONNECT");
				break;
			case DROPOUT:
				printf("DROPOUT    ");
				break;
			case POOL:
				printf("POOLING    ");
				break;
		}
		printf(" (%  4d,%  4d,%  4d)\n", cnn->camadas[i]->saida->x, cnn->camadas[i]->saida->y,
		       cnn->camadas[i]->saida->z);
	}
	printf("ESTA CORRETO? (S/N)");
	int c = toupper(getchar());
	if (c == 'N') {

		erro = -5;
		goto end;
	}
	cnn->flags = CNN_FLAG_CALCULE_ERROR;
	if (!SetCurrentDirectory(p.home)) {
		fprintf(stderr, "Falha ao mudar para o diretorio %s\n", p.home);
		erro = 2;
		goto end;
	}

	erro = loadSamples(cnn, &input, &target, &targeti, p.arquivoContendoImagens, p.arquivoContendoRespostas,
	                   p.Numero_Classes, p.Numero_Imagens,
	                   p.bytes_remanessentes_imagem, p.bytes_remanessentes_classes);
	if (erro)goto end;
	train(cnn, input, target, targeti, p.Numero_epocas, p.SalvarBackupACada, p.Numero_ImagensTreino,
	      p.estatisticasDeTreino);
	snprintf(buff,300,"%s.cnn",p.nome);
	FILE *fileCnn = fopen(buff,"wb");
	cnnSave(cnn,fileCnn);
	fclose(fileCnn);

	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->entrada->x * cnn->camadas[cnn->size - 1]->entrada->y *
	                    cnn->camadas[cnn->size - 1]->entrada->z;
	fitness(cnn, input + p.Numero_ImagensTreino * inputSize, targeti + p.Numero_ImagensTreino,
	        p.Numero_Classes, p.names, p.Numero_ImagensAvaliacao, p.SalvarSaidasComoPPM, p.nome,
	        p.estatiscasDeAvaliacao);

	printf("\ntreino terminado\n");

	end:
	if (input)free(input);
	if (target)free(target);
	if (targeti)free(targeti);
	if (cnn) {
		releaseCnn(&cnn);
	}
	if (p.names)free(p.names);
	printf("\n");
	system("pause");
	return erro;
}

