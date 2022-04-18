#define UTF8

#include "windows.h"
#include <stdio.h>
#include <cnn/cnn_lua.h>
#include <png/png.h>
#include <io.h>
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

#define  TMP_FILE_NAME_ARCH "edit_archtmp_lua.lua"

void on_end_epoca(const Setup self, int epoca) {
	if (!GUI.avaliar) {
		return;
	}
//	double t0 = seconds();
	GUI.avaliando = 1;
	float win, custo;
	self->fast_fitnes(self, &win, &custo);
	self->itrain.faval = -1;
	Axe *ax_erro = GUI.figs[0].getAxe(GUI.figs, 1);
	Axe *ax_win = GUI.figs[0].getAxe(GUI.figs + 1, 3);
	ax_erro->pushDraw(ax_erro, epoca, custo);
	ax_win->pushDraw(ax_win, epoca, win);
	self->iteste.mse = custo;
	self->iteste.winRate = win;
	printf("epoca %d winrate %.2f%% ecx %f\n", epoca, win, custo);
	GUI.avaliando = 0;
//	self->itrain.t0 += seconds() - t0;
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->cnn->ecx)
}

int fileExist(char *filename) {
	return access(filename, F_OK) == 0;
}

#include "locale.h"

int cnnMain(int nargs, char **args) {
	char luaFile[250] = {0};
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
	printf("\n\n");
	s->on_endEpoca = (void (*)(const struct Setup_t *, int)) on_end_epoca;

	if (s->ok(s)) {
		s->iLoad.t0 = seconds(); // captura o tempo incial
		s->runing = 1; // informa que está rodando
		hload = Thread_new(s->loadImagens, s); // inicia thread hload
		GUI.make_loadImages();
		while (s->runing) { // enquanto a thread hload estiver rodando, (é mais eficiente que Thread_IsAlive(hload)
			GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - s->iLoad.t0);
			Sleep(100);
		}
		GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - s->iLoad.t0);
		Thread_Release(hload);
		s->iLoad.t0 = seconds() - s->iLoad.t0;
		printf("Tempo para leitura de imagens %.3lf , imagens por segundo %.2lf\n", s->iLoad.t0, s->iLoad.imTotal / s->iLoad.t0);
	}
	if (s->ok(s)) {
		s->iLoad.t0 = seconds();
		s->runing = 1;
		hload = Thread_new(s->loadLabels, s);
		GUI.make_loadLabels();
		while (s->runing) {
			GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - s->iLoad.t0);
			Sleep(100);
		}
		GUI.updateLoadImagens(s->iLoad.imAtual, s->iLoad.imTotal, seconds() - s->iLoad.t0);
		showCursor(1);
		Thread_Release(hload);
		s->iLoad.t0 = seconds() - s->iLoad.t0;
		printf("Tempo para leitura de labels %.3lf s, imagens por segundo %.2lf\n", s->iLoad.t0, s->iLoad.imTotal / s->iLoad.t0);
	}
	/// TREINO
	if (s->ok(s)) {
		Itrain treino;
		s->itrain.t0 = seconds();
		s->runing = 1;
		GUI.make_train();
		while (!GUI.endDraw);
		if (s->cnn->cm[s->cnn->l - 1]->layer_id == FULLCONNECT_ID && ((CamadaFullConnect) s->cnn->cm[s->cnn->l - 1])->fa.id == FSOFTMAX) {
			GUI.figs[0].title = "Cross-Entropy";
		}
		if (s->useBatch) {
			htreino = Thread_new(s->treinarBatch, s);
			GUI.setText(GUI.status, "Treinando Batch size %lld", s->batchSize);
		} else {
			htreino = Thread_new(s->treinar, s);
		}


		while (s->runing) {
			treino = s->itrain;
			GUI.updateTrain(treino.faval, treino, seconds() - treino.t0);
			Sleep(100);
		}
		Thread_Release(htreino);
		treino = s->itrain;
		GUI.updateTrain(treino.faval, treino, seconds() - treino.t0);
		printf("Tempo para treino %.3lf s\n", seconds() - treino.t0);
		Sleep(10);
		GUI.capture(s->treino_out);
		ECX_IF_FAILED(s->cnn->ecx, end)
		char buff[250] = "";
		int i = 0;
		while (1) {
			i++;
			snprintf(buff, 250, "resultados/%s(%d).gabph", s->nome, i);
			if (!fileExist(buff)) {
				break;
			}
		}
		// verifica se arquivo existe

		// erro_treino, erro_avaliado
		// win_treino, win_avaliado

		FILE *f = fopen(buff, "wb");
		printf("%s\n",buff);
		saveaxe("Erro treino", GUI.figs[0].axes, f);
		saveaxe("Erro avaliacao", GUI.figs[0].axes + 1, f);
		saveaxe("winrate treino", GUI.figs[1].axes + 0, f);
		saveaxe("winrate avaliacao", GUI.figs[1].axes + 3, f);
		s->cnn->fprint(s->cnn, f, "#");
		fclose(f);
	}

	char nome[250];
	goto end;
	snprintf(nome, 250, "%s.cnn", s->nome);
	GUI.setText(GUI.status, "Salvando cnn em %s", nome);
	s->cnn->save(s->cnn, nome);
	/// FITNESS
	double t0;
	if (s->ok(s)) {
		t0 = seconds();
		s->runing = 1;
		hteste = Thread_new(s->avaliar, s);
		Iteste teste;
		GUI.make_teste();
		while (s->runing) {
			teste = s->iteste;
			GUI.updateTeste(teste.imAtual, teste.imTotal, teste.mse, teste.meanwinRate, seconds() - t0);
			Sleep(100);
		}
		GUI.updateTeste(teste.imAtual, teste.imTotal, teste.mse, teste.winRate, seconds() - t0);
		Thread_Release(hteste);
		t0 = seconds() - t0;
		printf("Tempo para treino %.3lf s\n", t0);
		GUI.capture(s->teste_out);
		s->saveStatistic(s);
	}

	char *tmp = asprintf(NULL, "Custo treino %lf\n"
							   "win hate treino %lf.\n"
							   "Custo avaliacao %lf\n"
							   "win hate teste %lf.\n"
							   " Deseja mudar a arquitetura da rede?", s->itrain.mse, s->itrain.winRateMedioep, s->iteste.mse, s->iteste.winRate);

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
	ECX_REGISTRE_FUNCTION_IF_ERROR(s->cnn->ecx)
	// salvar estatisticas
	if (!s->cnn->ecx->error) {
		if(!fileExist("endtrain.py")){
			FILE *py = fopen("endtrain.py","w");
			fprintf(py,"from gtts import gTTS\n"
					   "from playsound import playsound\n"
					   "import sys\n"
					   "if len(sys.argv) != 2:\n"
					   "    arg = 'O treinamento terminou'\n"
					   "else:\n"
					   "    arg = sys.argv[1]\n"
					   "gTTS(arg,lang =\"pt\").save('sample.mp3')\n"
					   "playsound('sample.mp3')");
			fclose(py);
		}
		setlocale(LC_ALL, "Portuguese");
		char *vtmp = asprintf(NULL, "python endtrain.py \"O treino terminou. Custo treino %.4g . "
									"acertos treino %.1f por cento."
									"Custo avaliação %.4g. "
									"acertos avaliação %.1f por cento.\"", s->itrain.mse, s->itrain.winRate, s->iteste.mse, s->iteste.winRate);

		system(vtmp);
		gab_free(vtmp);
	} else {
		s->cnn->ecx->print(s->cnn->ecx);
	}
	return s->release(&s);
}
