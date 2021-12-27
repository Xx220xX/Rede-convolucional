#include "windows.h"
#include <stdio.h>
#include <cnn/cnn_lua.h>
#include <png/png.h>
#include "ui_win_api/ui.h"
#include "setup/Setup.h"
#include "conio2/conio2.h"
#include "thread/Thread.h"
#include "camadas/all_camadas.h"

void Cnn_asimage(Cnn c, char *file, int largura, int altura, ...) {
	Tensor s = NULL;
	va_list v;
	uint8_t *image = gab_alloc(altura, largura);
	va_start(v, altura);
	s = va_arg(v, Tensor);
	int nTensors = 0;
	while (s) {
		nTensors++;
		s = va_arg(v, Tensor);
	}
	va_end(v);
	va_start(v, altura);
	int h = altura / nTensors;
	int w;
	int sz;
	int k;
	for (int j = 0; j < nTensors; ++j) {
		s = va_arg(v, Tensor);
		w = largura / (s->z * s->w);
		sz = h < w ? h : w;
		for (int l = 0; l < s->w; ++l) {
			for (int z = 0; z < s->z; ++z) {
				k = l * s->z + z;
				if (s->x != 1) {
					s->imagegray(s, image, largura, altura, sz, sz, j * h, k * w, z, l);
				} else {
					s->imagegray(s, image, largura, altura, w, h, j * h, k * w, z, l);
				}
			}
		}
	}
	va_end(v);
	pngGRAY(file, image, largura, altura);
	gab_free(image);
}

/*
void on_train(const struct Setup_t *self, int label) {
	Cnn cnn = self->cnn;
	double  b = seconds();
	if(1/(b-t0)>5)return;
	t0 = b;

	Camada cm;
	static int printed = 0;
	for (int i = 0; i < cnn->l; ++i) {
		cm = cnn->cm[i];
		switch (cm->layer_id) {
			case CONVOLUCAO_ID:
				Client_sendTensor(CST_CONVOLUCAO(cnn, i)->W, i + 1);
				if(!printed){
					printf("CONVOLUCAO %d -> figure %d\n",i,i+1);
				}
				break;
			case CONVOLUCAOF_ID:
				Client_sendTensor(CST_CONVOLUCAOF(cnn, i)->W, i + 1);
				if(!printed){
					printf("CONVOLUCAOF %d -> figure %d\n",i,i+1);
				}
				break;
			case CONVOLUCAONC_ID:
				Client_sendTensor(CST_CONVOLUCAONC(cnn, i)->W,i+1);
				if(!printed){
					printf("CONVOLUCAONC %d -> figure %d\n",i,i+1);
				}
				break;
			case POOL_ID:
				break;
			case FULLCONNECT_ID:
				Client_sendTensor(CST_FULLCONNECT(cnn,i)->w,i+1);
				if(!printed){
					printf("FULLCONNECT W %d -> figure %d\n",i,i+1);
				}
//				Client_sendTensor(CST_FULLCONNECT(cnn,i)->b,j++);
//				if(!printed){
//					printf("FULLCONNECT B %d -> figure %d\n",i,j-1);
//				}
				break;
			case PADDING_ID:
				break;
			case DROPOUT_ID:
				break;
			case RELU_ID:

				break;
			case PRELU_ID:
//				Client_sendTensor(CST_PRELU(cnn,i)->A,i+1);
//				if(!printed){
//					printf("PRELU B %d -> figure %d\n",i,i+1);
//				}
				break;
			case SOFTMAX_ID:

				break;
			case BATCHNORM_ID:
//				Client_sendTensor(CST_BATCHNORM(cnn,i)->Y,i+1);
//				if(!printed){
//					printf("BATCHNORM Y %d -> figure %d\n",i,i+1);
//				}Client_sendTensor(CST_BATCHNORM(cnn,i)->B,i+1);
//				if(!printed){
//					printf("BATCHNORM B %d -> figure %d\n",i,i+1);
//				}
				break;
			default:
				fprintf(stderr,"Camada desconhecida, id %d\n",cm->layer_id);
		}
	}
//	Client_sendTensor(CST_CONVOLUCAO(cnn, 4)->filtros, j++);
//	Client_sendTensor(CST_CONVOLUCAO(cnn, 5)->filtros, j++);
	printed = 1;
//	Sleep(100);
}
*/
#define  TMP_FILE_NAME_ARCH "edit_archtmp_lua.lua"

int cnnMain(int nargs, char **args) {
	char luaFile[250] = {0};
	double t0;
	double im;
	HANDLE hload;
	HANDLE htreino;
	HANDLE hteste;
	Setup s = Setup_new();

	LCG_setSeed(time(0));

	GUI.can_run = &s->can_run;
	GUI.force_end = &s->force_end;
	getLuaFILE(luaFile, 250, nargs, (const char **) args);
	s->loadLua(s, luaFile);
	init:
	s->cnn->print(s->cnn, "--");
	if (s->ok(s)) {
		t0 = seconds(); // captura o tempo incial
		s->runing = 1; // informa que está rodando
		hload = Thread_new(s->loadImagens, s); // inicia thread hload
		GUI.make_loadImages();
		while (s->runing) { // enquanto a thread hload estiver rodando, (é mais eficiente que Thread_IsAlive(hload)
			GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - t0);
			Sleep(100);
		}
		GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - t0);
		Thread_Release(hload);
		t0 = seconds() - t0;
		printf("Tempo para leitura de imagens %.3lf , imagens por segundo %.2lf\n", t0, s->iLoad.imTotal / t0);
	}
	if (s->ok(s)) {
		t0 = seconds();
		s->runing = 1;
		hload = Thread_new(s->loadLabels, s);
		GUI.make_loadLabels();
		while (s->runing) {
			GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - t0);
			Sleep(100);
		}
		GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - t0);
		showCursor(1);
		Thread_Release(hload);
		t0 = seconds() - t0;
		printf("Tempo para leitura de labels %.3lf s, imagens por segundo %.2lf\n", t0, s->iLoad.imTotal / t0);
	}
	if (s->ok(s)) {
		t0 = seconds();
		s->runing = 1;
		if(s->useBatch) {
			htreino = Thread_new(s->treinarBatch, s);
		}else{
			htreino = Thread_new(s->treinar, s);
		}
		Itrain treino;
		GUI.make_train();
		while (s->runing) {
			treino = s->itrain;
			GUI.updateTrain(treino.imAtual, treino.imTotal, treino.epAtual, treino.epTotal, treino.mse, treino.winRate, seconds() - t0);
			Sleep(100);
		}
		Thread_Release(htreino);
		treino = s->itrain;
		GUI.updateTrain(treino.imAtual, treino.imTotal, treino.epAtual, treino.epTotal, treino.mse, treino.winRate, seconds() - t0);
		t0 = seconds() - t0;
		printf("Tempo para treino %.3lf s\n", t0);
		GUI.capture(s->treino_out);
	}
	char nome[250];
	snprintf(nome, 250, "%s.cnn", s->nome);
	GUI.setText(GUI.status, "Salvando cnn em %s", nome);
	s->cnn->save(s->cnn, nome);
	if (s->ok(s)) {
		t0 = seconds();
		s->runing = 1;
		hteste = Thread_new(s->avaliar, s);
		Iteste teste;
		GUI.make_teste();
		while (s->runing) {
			teste = s->iteste;
			GUI.updateTeste(teste.imAtual, teste.imTotal, teste.mse, teste.winRate, seconds() - t0);
			Sleep(100);
		}
		GUI.updateTeste(teste.imAtual, teste.imTotal, teste.mse, teste.winRate, seconds() - t0);
		Thread_Release(htreino);
		t0 = seconds() - t0;
		printf("Tempo para treino %.3lf s\n", t0);
		GUI.capture(s->teste_out);
		s->saveStatistic(s);
	}

	char *tmp = asprintf(NULL, "MSE %lf\nwin hate %lf.\n Deseja mudar a arquitetura da rede?", s->itrain.mse, s->itrain.winRate);
	if (!s->cnn->ecx->error && dialogBox("Treino terminado!", tmp)) {
		gab_free(tmp);
		s->force_end = 0;
		s->can_run = 1;
		s->cnn->ecx->error = 0;
		// criar arquivo temporario
		FILE *tmpf = fopen(TMP_FILE_NAME_ARCH, "w");
		// copiar help
		helpCnn(tmpf, "-- ");
		//copiar arquitetura
		s->cnn->fprint(s->cnn, tmpf, "--");
		fclose(tmpf);
		// limpar cnn
		while (s->cnn->l > 0) {
			s->cnn->removeLastLayer(s->cnn);
		}
		{
			STARTUPINFO si;
			PROCESS_INFORMATION pi;

			ZeroMemory(&si, sizeof(si));
			si.cb = sizeof(si);
			ZeroMemory(&pi, sizeof(pi));

			// Start the child process.
			if (!CreateProcess(NULL,   // No module name (use command line)
							   "notepad "TMP_FILE_NAME_ARCH,        // Command line
							   NULL,           // Process handle not inheritable
							   NULL,           // Thread handle not inheritable
							   FALSE,          // Set handle inheritance to FALSE
							   0,              // No creation flags
							   NULL,           // Use parent's environment block
							   NULL,           // Use parent's starting directory
							   &si,            // Pointer to STARTUPINFO structure
							   &pi)           // Pointer to PROCESS_INFORMATION structure
					) {
				printf("CreateProcess failed (%lu).\n", GetLastError());
				goto end;
			}

			// Wait until child process exits.
			WaitForSingleObject(pi.hProcess, INFINITE);

			// Close process and thread handles.
			CloseHandle(pi.hProcess);
			CloseHandle(pi.hThread);
		}
		CnnLuaLoadFile(s->cnn, TMP_FILE_NAME_ARCH);
		remove(TMP_FILE_NAME_ARCH);
		if (s->ok(s)) {
			s->epoca_atual = 0;
			s->imagem_atual_teste = 0;
			s->imagem_atual_treino = 0;
			goto init;
		}
	}

	end:
	return s->release(&s);
}
