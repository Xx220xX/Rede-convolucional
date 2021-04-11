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


int loadSamples(Cnn cnn, double **images, double **labels, unsigned char **labelsI, char *imageFile, char *labelFile,
                size_t numberOfLabels, size_t numberOfSamples, size_t remainImage, size_t remainLabel) {
	int err = 0;
	size_t pixelsByImage = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	int numeroClasse = cnn->camadas[cnn->size - 1]->entrada->x * cnn->camadas[cnn->size - 1]->entrada->y *
	                   cnn->camadas[cnn->size - 1]->entrada->z;

	size_t samples;
	*images = (double *) calloc(sizeof(double), pixelsByImage * numberOfSamples);
	*labels = (double *) calloc(sizeof(double), numberOfLabels * numberOfSamples);
	*labelsI = (unsigned char *) calloc(sizeof(unsigned char), numberOfLabels * numberOfSamples);
	FILE *fimage = fopen(imageFile, "rb");
	if (!fimage){fprintf(stderr,"Imagens nao foram encontradas em %s\n",imageFile);err = -1;goto error;}
	fread(*images, 1, remainImage, fimage);// bytes remanessentes de cabeçalho
	normalizeImage(*images, numberOfSamples
	                        * pixelsByImage, cnn->cl, cnn->queue, cnn->kerneldivInt, fimage, &samples);
	fclose(fimage);

	if (numberOfSamples * pixelsByImage != samples) {
		err = -2;
		goto error;
	}

	FILE *flabel = fopen(labelFile, "rb");
	if (!fimage){fprintf(stderr,"Labels nao foram encontradas em %s\n",labelFile);err = -1;goto error;}
	fread(*labels, 1, remainLabel, fimage);
	loadTargetData(*labels, *labelsI, numeroClasse,numberOfSamples,  cnn->cl, cnn->queue, cnn->kernelInt2Vector, flabel,
	               &samples);
	fclose(flabel);

	if (numberOfSamples != samples) {
	fprintf(stderr,"Smp %lld lidos %lld\n",numberOfSamples,samples);
		err = -3;
		goto error;
	}

	error:
	if (err) {
		free(*images);
		free(*labels);
		free(*labelsI);
		*images = *labels = NULL ;
		*labelsI = NULL;
		fprintf(stderr, "Error while try read data\n");
		return err;
	}
	return 0;
}

int train(Cnn cnn, double *images, double *labels, unsigned char *labelsI, int epocs, int saveCNN, int samples,
          char *outputMDTable) {
	printf("here\n");
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
			snprintf(buff, 250, "/imgs/%s%d.ppm",name, caso + 1);
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
		fprintf(f, "| %s | %d | %d | %g |\n", names[i].names, numeroCasosOcorridos[i], acertosPorClasse[i],
		        erroPorClasse[i]);
	}
	fclose(f);
}

int loadLuaParameters(char *luaFile, ParametrosCnnALL *p) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	loadCnnLuaLibrary(L);
	printf("%s\n", luaFile);
	luaL_loadfile(L, luaFile);
	int error = lua_pcall(L, 0, 0, 0);

	// o scrip foi executado
	if (error) {
		fprintf(stderr, "stack:%d\n", lua_gettop(L));
		fprintf(stderr, "erro:%d\n", error);
		fprintf(stderr, "message:%s\n", lua_tostring(L, -1));
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
	return 0;
}

int main(int nargs, char **args) {
	Cnn cnn = NULL;
	globalcnn = &cnn;
	char kernelFile[] = "../kernels/gpu_function.cl";
	globalKernel = kernelFile;
	char path[] = "..";
	char buff[300];
	char luaFile[] = "TESTE_NUMEROS_0_9.lua";
	snprintf(buff, 300, "%s/%s", path, luaFile);
	ParametrosCnnALL p = {0};


	int erro = loadLuaParameters(buff, &p);
	double *input = NULL, *target = NULL;
	unsigned char *targeti = NULL;
	if (erro )goto end;
	if (!cnn) {fprintf(stderr,"Nao foi encontrado uma arquitetura de rede"); goto end; }

	if(!SetCurrentDirectory(p.home)){
		fprintf(stderr,"Falha ao mudar para o diretorio %s\n",p.home);
		erro = 2;
		goto end;
	}

	erro = loadSamples(cnn, &input, &target, &targeti, p.arquivoContendoImagens, p.arquivoContendoRespostas,
	                   p.Numero_Classes, p.Numero_Imagens,
	                   p.bytes_remanessentes_imagem, p.bytes_remanessentes_classes);
	if (erro )goto end;
	train(cnn,input,target,targeti,p.Numero_epocas,p.SalvarBackupACada,p.Numero_ImagensTreino,p.estatisticasDeTreino);

	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->entrada->x * cnn->camadas[cnn->size - 1]->entrada->y *
	                    cnn->camadas[cnn->size - 1]->entrada->z;
	fitness(cnn,input+p.Numero_ImagensTreino*inputSize,targeti+p.Numero_ImagensTreino,
		 p.Numero_Classes,p.names,p.Numero_ImagensAvaliacao,p.SalvarSaidasComoPPM,p.nome,p.estatiscasDeAvaliacao);

	end:
	if (input)free(input);
	if (target)free(target);
	if (targeti)free(targeti);
	if (cnn) {
//		printf("cnn foi carregada\n");
		releaseCnn(&cnn);
	}
	if (p.names)free(p.names);
	getchar();
	return erro;
}

