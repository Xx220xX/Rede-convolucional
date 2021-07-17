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
#include "preparaImagens.h"

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
	size_t initTime = getms();
	size_t initTimeAll = getms();
	double erro = 0.0;
	double erroMedio = 0.0;

	// inicializa infomações de treino utilizado pela thread ui
	InfoTrain info = {samples, key, acertos, epoca, epocs, initTime, initTimeAll, erroMedio};
	info.finish = 1;
	// arquivo da tabela
	SalveJsArgs jsargs = {0};
	SalveJsArgs aux;
	jsargs.jsAcerto = fopen("js/dataAcerto.js", "w");
	jsargs.jsErro = fopen("js/dataErro.js", "w");
	jsargs.jsEpoca = fopen("js/dataEpoca.js", "w");
	fprintf(jsargs.jsAcerto, "var acerto = [");
	fprintf(jsargs.jsErro, "var normaErro = [");
	fprintf(jsargs.jsEpoca, "var epoca = [");
	jsargs.len = samples;

	// obtem tamanho de entrada e saída da rede
	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->saida->x * cnn->camadas[cnn->size - 1]->saida->y *
						cnn->camadas[cnn->size - 1]->saida->z;


	int r;
	int stop = 0;
	int *index = (int *) calloc(samples, sizeof(int));
//	cnn->camadas[cnn->size-1]->setLearn(cnn->camadas[0],0);
//	cnn->camadas[cnn->size-1]->setLearn(cnn->camadas[cnn->size-1],0);
	for (int i = 0; i < samples; i++)index[i] = i;
	// inicia treinamento
	for (; epoca < epocs && !cnn->error.error; epoca++) {
		if (stop == 'q')break;
		initTime = getms();
		jsargs.acerto = calloc(jsargs.len, sizeof(double));
		jsargs.erro = calloc(jsargs.len, sizeof(double));
		jsargs.epoca = calloc(jsargs.len, sizeof(double));
//		LCG_shuffle(index, samples, sizeof(int));
		erro = 0;
		erroMedio = 0;
		acertos = 0;
		for (key = 0; key < samples && !cnn->error.error; key++) {

			caso = index[key];
			if (kbhit() && (stop = tolower(getch())) == 'q')break;
			CnnCall(cnn, images + inputSize * caso);
//			printf("call %d\n",cnn->error.error);
			CnnLearn(cnn, labels + outputSize * caso);
//			printf("learn %d\n",cnn->error.error);
			CnnCalculeError(cnn);
			r = CnnGetIndexMax(cnn);
//			printf("erro[%d] %lf",cnn->size,cnn->normaErro);
//			double er;
//			for(int l=0;l<cnn->size;l++){
//				printf("%d ",l);
//				printf(" 0x%llX\n",(long long int)((CamadaLearnable)cnn->camadas[l])->dw);
//				TensorGetNorm(cnn->queue,((CamadaLearnable)cnn->camadas[l])->dw,&er);
//				printf("\n%lf",er);
//
//				printf("\n0x%llX ",(long long int)cnn->camadas[l]->gradsEntrada);
//				TensorGetNorm(cnn->queue,cnn->camadas[l]->gradsEntrada,&er);
//				printf("\n%lf",er);
//
//				printf("\n0x%llX ",((CamadaLearnable)cnn->camadas[l])->w);
//				TensorGetNorm(cnn->queue,((CamadaLearnable)cnn->camadas[l])->w,&er);
//				printf("\n%lf)",er);
//				printf("\n\n\n");
//
//			}
//			printf("\n");
			if (r == labelsI[caso]) { acertos += 1; }

			erro += cnn->normaErro;
			erroMedio = erro / (key + 1);
			jsargs.erro[key] = erroMedio;
			jsargs.epoca[key] = epoca + key / ((double) samples);
			jsargs.acerto[key] = acertos * 100.0 / ((double) key + 1.0);

			if (info.finish && !cnn->error.error) {
				info.erro = erroMedio;;
				info.acertos = acertos;
				info.msInitEpoca = initTime;
				info.imagemAtual = key;
				info.epoca = epoca;
				info.finish = 0;
				pthread_create(NULL, NULL, (void *(*)(void *)) showInfoTrain, &info);
			}
		}
		pthread_join(jsargs.tid, NULL);

		aux = jsargs;
		aux.len = key;
		pthread_create(&jsargs.tid, NULL, (void *(*)(void *)) salveJS, &aux);
	}
	printf(" salvando dados\n");
	pthread_join(jsargs.tid, NULL);
	fprintf(jsargs.jsAcerto, "];\nplot('graficoAcerto',epoca,acerto,'epoca','acertos','Acertos durante treino')\n");
	fprintf(jsargs.jsErro, "];\nplot('graficoErro',epoca,normaErro,'epoca','erro','Erro quadratico durante treino')\n");
	fprintf(jsargs.jsEpoca, "];\n");
	fclose(jsargs.jsAcerto);
	fclose(jsargs.jsErro);
	fclose(jsargs.jsEpoca);
	info.finish = 1;
	// finalizando ui
	while (!info.finish) { Sleep(1); }

	return cnn->error.error;
}

int
fitness(Cnn cnn, double *images, unsigned char *labelsI, int nClass, Nomes *names, int samples, size_t imagesSaveOutput,
		char *name, char *outputMDTable) {
	int caso = 0;
	int acertos = 0;
	int *acertosPorClasse = calloc(sizeof(int), nClass);
	int *numeroCasosOcorridos = calloc(sizeof(int), nClass);
	size_t initTime = getms();
	InfoTeste info = {nClass, samples, caso, acertos, initTime, acertosPorClasse, numeroCasosOcorridos,
					  names, 20, 1};

	pthread_create(NULL, NULL, (void *(*)(void *)) showInfoTest, (void *) &info);
	int r;
	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	for (caso = 0; caso < samples; caso++) {
		if (kbhit() && tolower(getch()) == 'q')break;
		CnnCall(cnn, images + inputSize * caso);
		r = CnnGetIndexMax(cnn);
		numeroCasosOcorridos[labelsI[caso]]++;
		if (r == labelsI[caso]) {
			acertosPorClasse[r]++;
			acertos++;
		}
		if (info.finish) {
			info.imagemAtual = caso;
			info.acertos = acertos;
			info.finish = 0;
			pthread_create(NULL, NULL, (void *(*)(void *)) showInfoTest, &info);
		}
	}
	FILE *js_tabela;
	js_tabela = fopen("js/fitnes.js", "w");
	fprintf(js_tabela, "tablePutColum('tabela_fitnes',['Classe','acertos','total','porcentagem']);");
	for (int i = 0; i < 10; i++) {
		fprintf(js_tabela, "tablePutColum(tabela_fitnes,['%s' , %d , %d , '%.2lf%%']);\n", names[i].names,
				numeroCasosOcorridos[i], acertosPorClasse[i],
				acertosPorClasse[i] * 100.0 / (double) (numeroCasosOcorridos[i] + 1e-14));
	}
	fclose(js_tabela);
	while (!info.finish) { Sleep(1); }
	return cnn->error.error;
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
	printf("script carregado\n");
	// o scrip foi executado
	if (error) {
		fprintf(stderr, "Falha ao carregar scrip\n");
		fprintf(stderr, "stack%d\n", lua_gettop(L));
		fprintf(stderr, "erro%d\n", error);
		fprintf(stderr, "message%s\n", lua_tostring(L, -1));
		system("pause");
		return error;
	}

	printf("carregando informações\n");
	// verificar se as variaveis foram setadas
	char *tmp;

	GETLUAVALUE(p->Numero_epocas, L, "Numero_epocas", integer,
				printf("ERRO Numero_epocas não foi atribuído");return -1;);
	GETLUAVALUE(globalcnn[0]->parametros.hitLearn, L, "taxaAprendizado", number,
				printf("warning taxaAprendizado não foi atribuído, será considerado como 0.1\n"););
	GETLUAVALUE(globalcnn[0]->parametros.decaimentoDePeso, L, "decaimentoDePeso", number,
				printf("warning decaimentoDePeso não foi atribuído, será considerado como 0.0\n"););
	GETLUAVALUE(globalcnn[0]->parametros.momento, L, "momento", number,
				printf("warning momento não foi atribuído, será considerado como 0.0\n"););
	GETLUAVALUE(p->Numero_Imagens, L, "Numero_Imagens", integer,
				printf("ERRO Numero_Imagens não foi atribuído");return -1;);
	GETLUAVALUE(p->Numero_ImagensAvaliacao, L, "Numero_ImagensAvaliacao", integer,
				printf("ERRO Numero_ImagensAvaliacao nao foi atribuído");return -1;);
	GETLUAVALUE(p->Numero_ImagensTreino, L, "Numero_ImagensTreino", integer,
				printf("ERRO Numero_ImagensTreino não foi atribuído");return -1;);
	GETLUAVALUE(p->SalvarSaidasComoPPM, L, "SalvarSaidasComoPPM", integer,
				printf("ERRO SalvarSaidasComoPPM não foi atribuído");return -1;);
	GETLUAVALUE(p->SalvarBackupACada, L, "SalvarSaidasComoPPM", integer,
				printf("ERRO SalvarSaidasComoPPM não foi atribuído");return -1;);
	GETLUAVALUE(p->Numero_Classes, L, "Numero_Classes", integer,
				printf("ERRO Numero_Classes não foi atribuído");return -1;);
	GETLUAVALUE(p->bytes_remanessentes_classes, L, "bytes_remanessentes_classes", integer,
				printf("ERRO bytes_remanessentes_classes não foi atribuído");return -1;);
	GETLUAVALUE(p->bytes_remanessentes_imagem, L, "bytes_remanessentes_imagem", integer,
				printf("ERRO bytes_remanessentes_imagem não foi atribuído");return -1;);

	GETLUASTRING(p->nome, tmp, MAX_STRING_LEN, L, "nome", printf("ERRO nome não foi atribuído");return -1;);
	GETLUASTRING(p->home, tmp, MAX_STRING_LEN, L, "home", printf("ERRO home não foi atribuído");return -1;);

	GETLUASTRING(p->estatisticasDeTreino, tmp, MAX_STRING_LEN, L, "estatisticasDeTreino",
				 printf("ERRO estatisticasDeTreino não foi atribuído");return -1;);
	GETLUASTRING(p->estatiscasDeAvaliacao, tmp, MAX_STRING_LEN, L, "estatiscasDeAvaliacao",
				 printf("ERRO estatiscasDeAvaliacao não foi atribuído");return -1;);
	GETLUASTRING(p->arquivoContendoImagens, tmp, MAX_STRING_LEN, L, "arquivoContendoImagens",
				 printf("ERRO arquivoContendoImagens não foi atribuído");return -1;);
	GETLUASTRING(p->arquivoContendoRespostas, tmp, MAX_STRING_LEN, L, "arquivoContendoRespostas",
				 printf("ERRO arquivoContendoRespostas não foi atribuído");return -1;);

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



