//#define LOG_CNN_KERNELCALL
#include <conio.h>
#include <ctype.h>
#include <windows.h>
#include "../cnn.h"
#include "uteisTreino.h"
#include"conioHUD.h"
#include "../lua/lua.h"
#include "../lua/lualib.h"
#include "../lua/lauxlib.h"
#include "CnnLua.h"
#include<dirent.h>

void showDir() {
	DIR *dir;
	struct dirent *ent;
	char currentDir[250] = {0};
	GetCurrentDirectory(250, currentDir);
	printf("Current dir%s\n", currentDir);
	/*if ((dir = opendir(currentDir)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	getchar();*/
}
/**
 *  Carrega as imagens do arquivo
 * @param cnn  instancia valida de uma cnn (para utilizar a gpu)
 * @param images  ponteiro para vetor double onde as imagens serão salvas
 * @param remainImage  bits de cabeçalho
 * @param numberOfSamples  numero de imagens a ser lida
 * @param imageFile  string contendo o nome do arquivo que contem as imagens
 * @return  0 caso sucesso. -1 caso não seja possivel abrir o arquivo. -2 caso o numero de imagens
 * lidas seja diferente do numero de imagens especificado
 */
int loadImage(Cnn cnn, double **images, size_t remainImage, size_t numberOfSamples, char *imageFile) {
	// obtem o tamanho de cada imagem
	size_t pixelsByImage = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t samples;
	// abre o arquivo em modo leitura
	FILE *fimage = fopen(imageFile, "rb");
	if (!fimage) {
		fprintf(stderr, "Imagens nao foram encontradas em %s\n", imageFile);
		*images = NULL;
		return -1;
	}

	// aloca memoria para instanciar imagens
	*images = (double *) calloc(sizeof(double), pixelsByImage * numberOfSamples);
	// faz a leitura dos bytes remanecentes
	fread(*images, 1, remainImage, fimage);// bytes remanessentes de cabeçalho

	// normaliza imagens antes de 0 a 255 para 0 a 1 (utilizando a GPU)
	normalizeImage(*images, numberOfSamples * pixelsByImage,
	               cnn->cl, cnn->queue, cnn->kerneldivInt, fimage, &samples);
// fecha o arquivo
	fclose(fimage);
	// verifica se a leitura foi correta
	if (numberOfSamples * pixelsByImage != samples) {
		fprintf(stderr, "As imagens nao foram lidas corretamente\n");
		free(*images);
		*images = NULL;
		return -2;
	}
	return 0;

}

/**
 *  Carrega as respostas do arquivo
 * @param cnn  instancia valida de uma cnn (para utilizar a gpu)
 * @param labels  ponteiro para vetor double onde as respostas serão salvas
 * @param remainLabel  bits de cabeçalho
 * @param numberOfSamples  numero de respostas a ser lida
 * @param numeroSaidas  numero de classes total
 * @param labelFile  string contendo o nome do arquivo que contem as respostas
 * @return  0 caso sucesso. -1 caso não seja possivel abrir o arquivo. -2 caso o numero de respostas
 * lidas seja diferente do numero de respostas especificado
 */
int loadLabel(Cnn cnn, double **labels, unsigned char **labelsI, size_t remainLabel, size_t numberOfSamples,
              size_t numeroSaidas, char *labelFile) {
	// abre o arquivo em modo leitura
	FILE *flabel = fopen(labelFile, "rb");
	if (!flabel) {
		fprintf(stderr, "Labels nao foram encontradas em %s\n", labelFile);
		*labels = NULL;
		*labelsI = NULL;
		return -1;
	}
	// aloca memoria para os vetores de resposta
	*labels = (double *) calloc(sizeof(double), numeroSaidas * numberOfSamples);
	// aloca memoria para as respostas (modo numerico)
	*labelsI = (unsigned char *) calloc(sizeof(unsigned char), numberOfSamples);
	// faz a leitura dos bytes remanecentes
	fread(*labels, 1, remainLabel, flabel);

	size_t lidos = 0;
	// chama função para converter de modo numerico para modo vetor 
	loadTargetData(*labels, *labelsI, numeroSaidas, numberOfSamples, cnn->cl, cnn->queue, cnn->kernelInt2Vector, flabel,
	               &lidos);
	// fecha o arquivo
	fclose(flabel);
	// verifica se o numero de respostas lidas está correto
	if (numberOfSamples != lidos) {
		fprintf(stderr, "Esperado %lld, lidos %lld\n", numberOfSamples, lidos);
		*labels = NULL;
		*labelsI = NULL;
		return -2;
	}
	return 0;
}

/**
 * Carrega as imagens e as respostas 
 * @param cnn   instancia valida de uma cnn (para utilizar a gpu)
 * @param images ponteiro para vetor double onde as imagens serão salvas
 * @param labels ponteiro para vetor double onde as respostas serão salvas
 * @param labelsI ponteiro para vetor char onde as respostas serão salvas
 * @param imageFile string contendo o nome do arquivo que contem as imagens
 * @param labelFile string contendo o nome do arquivo que contem as respostas
 * @param numberOfLabels numero de classes possiveis
 * @param numberOfSamples numero de exemplos a ser carregado
 * @param remainImage  bytes de cabeçalho para imagens
 * @param remainLabel bytes de cabeçalho para respostas
 * @return caso nao contenha erro retorna 0,caso falhe na leitura das imagens retorna -8&erro, caso falhe na leitura das respostas retorna -16&erro
 */
int loadSamples(Cnn cnn, double **images, double **labels, unsigned char **labelsI, char *imageFile, char *labelFile,
                size_t numberOfLabels, size_t numberOfSamples, size_t remainImage, size_t remainLabel) {

	int erro = 0;
	// le imagens
	if ((erro = loadImage(cnn, images, remainImage, numberOfSamples, imageFile)))return 8 & erro;
	// le respostas
	if ((erro = loadLabel(cnn, labels, labelsI, remainLabel, numberOfSamples, numberOfLabels, labelFile)))
		return 16 & erro;
	return 0;
}

/**
 * Salva imagem no formato ppm. Seu nome é sua resposta e contém tmb o vetor correspondente
 * @param images vetor de imagens
 * @param labels vetor de respostas em modo vetor
 * @param labelsI vetor de respostas em modo inteiro
 * @param n numero de imagens a ser salvas
 * @param x dimensao x da imagem (largura)
 * @param y dimensao y da imagem (altura)
 * @param z dimensao z da imagem
 * @param m numero de classes
 */
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

/**
 * Efetua o treino das imagens
 * @param cnn instancia valida da cnn
 * @param images imagens inicializadas e carregadas
 * @param labels respostas inicializadas e  carregadas modo vetor
 * @param labelsI respostas inicializadas e carregas modo inteiro
 * @param epocs numero de epocas total
 * @param  saveCNN #deprecated nao implementado
 * @param samples numero de exemplos para cada epoca
 * @param outputMDTable nome do arquivo de saida
 * @return retorna 0 caso ocorra tudo certo
 */
int train(Cnn cnn, double *images, double *labels, unsigned char *labelsI, int epocs, int saveCNN, int samples,
          char *outputMDTable) {
	int caso = 0;
	int key = 0;
	int acertos = 0;
//	printTestImages(images, labels, labelsI, samples, cnn->camadas[0]->entrada->x, cnn->camadas[0]->entrada->y,
//	                cnn->camadas[0]->entrada->z, cnn->camadas[cnn->size - 1]->saida->x);
//	system("pause");
	int epoca = 0;
	int threadInfoStop = 0;
	size_t initTime = getms();
	size_t initTimeAll = getms();
	double erro = 0.0;
	double erroMedio = 0.0;

	// inicializa infomações de treino utilizado pela thread ui
	InfoTrain info = {&samples, &key, &acertos, &epoca, &epocs, &threadInfoStop, &initTime, &initTimeAll, &erroMedio};
	pthread_t tid;
	// arquivo da tabela
	FILE *f;
	f = fopen(outputMDTable, "w");
	if (!f)return -1;
	fprintf(f, "\n|EPOCA | erro | acertos | tempo |\n|----|----|----|----|\n");
	// obtem tamanho de entrada e saída da rede
	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->saida->x * cnn->camadas[cnn->size - 1]->saida->y *
	                    cnn->camadas[cnn->size - 1]->saida->z;
	// incia thread ui
	pthread_create(&tid, NULL, (void *(*)(void *)) showInfoTrain, (void *) &info);
	int r;
	int stop = 0;
	// vetor para treinar com dados aleatorios
	int *index = (int *)calloc(samples,sizeof(int));
	// inicializa vetor index
	for(int i=0;i<samples;index[i] = i++);

	// inicia treinamento
	for (; epoca < epocs; epoca++) {
		if (stop == 'q')break;
		// captura o tempo em millissegundos
		initTime = getms();
		//mistura vetor de indices
		LCG_shuffle(index,samples,sizeof(int));

		erro = 0;
		erroMedio = 0;
		acertos = 0;
		fprintf(f, "|%d ", epoca);
		for (key = 0; key < samples; key++) {
			// pega o proximo exemplo
			caso = index[key];
			// verifica se o treino foi interrompido
			if (kbhit() && (stop = tolower(getch())) == 'q')break;
			// efetua o predict
			CnnCall(cnn, images + inputSize * caso);
			// efetua o backpropagation
			CnnLearn(cnn, labels + outputSize * caso);
			// retorna o indice do valor maximo da saida
			r = CnnGetIndexMax(cnn);
			//verifica acerto
			if (r == labelsI[caso]) { acertos += 1; }
			// soma o erro
			erro += cnn->normaErro;
			erroMedio = erro/(key+1);
		}
		fprintf(f, "| %g | %d | %llu |\n", erroMedio, acertos, (unsigned long long int) (getms() - initTime) * 1000);
	}
	fclose(f);
	//finaliza a thread ui
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
			snprintf(buff, 250, "imgs/%s_%d.ppm", name, caso + 1);
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
	fprintf(f, "| ---- | ---- | ---- | ---- |\n");
	for (int i = 0; i < 10; i++) {
		fprintf(f, "| %s | %d | %d | %g |\n", names[i].names, numeroCasosOcorridos[i], acertosPorClasse[i],
		        erroPorClasse[i]/(double)numeroCasosOcorridos[i]);
	}
	fclose(f);

}



int loadLuaParameters(char *luaFile, ParametrosCnnALL *p) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	loadCnnLuaLibrary(L);
//	printf("%s\n", luaFile);
	printf("carregando script");
//	printf(" %s", luaFile);
	printf("\n");
	//showDir();
	luaL_loadfile(L, luaFile);
	int error = lua_pcall(L, 0, 0, 0);

	// o scrip foi executado
	if (error) {
		fprintf(stderr, "Falha ao carregar scrip\n");
		fprintf(stderr, "stack%d\n", lua_gettop(L));
		fprintf(stderr, "erro%d\n", error);
		fprintf(stderr, "message%s\n", lua_tostring(L, -1));
		system("pause");
		return error;
	}
	// verificar se as variaveis foram setadas
	char *tmp;

	GETLUAVALUE(p->Numero_epocas, L, "Numero_epocas", integer,
	            printf("ERRO Numero_epocas nao foi atribuido");return -1;);
	GETLUAVALUE(globalcnn[0]->parametros.hitLearn, L, "taxaAprendizado", number,
	            printf("warning taxaAprendizado nao foi atribuido, será considerado como 0.1\n"););
	GETLUAVALUE(globalcnn[0]->parametros.decaimentoDePeso, L, "decaimentoDePeso", number,
	            printf("warning decaimentoDePeso nao foi atribuido, será considerado como 0.0\n"););
	GETLUAVALUE(globalcnn[0]->parametros.momento, L, "momento", number,
	            printf("warning momento nao foi atribuido, será considerado como 0.0\n"););
	GETLUAVALUE(p->Numero_Imagens, L, "Numero_Imagens", integer,
	            printf("ERRO Numero_Imagens nao foi atribuido");return -1;);
	GETLUAVALUE(p->Numero_ImagensAvaliacao, L, "Numero_ImagensAvaliacao", integer,
	            printf("ERRO Numero_ImagensAvaliacao nao foi atribuido");return -1;);
	GETLUAVALUE(p->Numero_ImagensTreino, L, "Numero_ImagensTreino", integer,
	            printf("ERRO Numero_ImagensTreino nao foi atribuido");return -1;);
	GETLUAVALUE(p->SalvarSaidasComoPPM, L, "SalvarSaidasComoPPM", integer,
	            printf("ERRO SalvarSaidasComoPPM nao foi atribuido");return -1;);
	GETLUAVALUE(p->SalvarBackupACada, L, "SalvarSaidasComoPPM", integer,
	            printf("ERRO SalvarSaidasComoPPM nao foi atribuido");return -1;);
	GETLUAVALUE(p->Numero_Classes, L, "Numero_Classes", integer,
	            printf("ERRO Numero_Classes nao foi atribuido");return -1;);
	GETLUAVALUE(p->bytes_remanessentes_classes, L, "bytes_remanessentes_classes", integer,
	            printf("ERRO bytes_remanessentes_classes nao foi atribuido");return -1;);
	GETLUAVALUE(p->bytes_remanessentes_imagem, L, "bytes_remanessentes_imagem", integer,
	            printf("ERRO bytes_remanessentes_imagem nao foi atribuido");return -1;);

	GETLUASTRING(p->nome, tmp, MAX_STRING_LEN, L, "nome", printf("ERRO nome nao foi atribuido");return -1;);
	GETLUASTRING(p->home, tmp, MAX_STRING_LEN, L, "home", printf("ERRO home nao foi atribuido");return -1;);

	GETLUASTRING(p->estatisticasDeTreino, tmp, MAX_STRING_LEN, L, "estatisticasDeTreino",
	             printf("ERRO estatisticasDeTreino nao foi atribuido");return -1;);
	GETLUASTRING(p->estatiscasDeAvaliacao, tmp, MAX_STRING_LEN, L, "estatiscasDeAvaliacao",
	             printf("ERRO estatiscasDeAvaliacao nao foi atribuido");return -1;);
	GETLUASTRING(p->arquivoContendoImagens, tmp, MAX_STRING_LEN, L, "arquivoContendoImagens",
	             printf("ERRO arquivoContendoImagens nao foi atribuido");return -1;);
	GETLUASTRING(p->arquivoContendoRespostas, tmp, MAX_STRING_LEN, L, "arquivoContendoRespostas",
	             printf("ERRO arquivoContendoRespostas nao foi atribuido");return -1;);

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



