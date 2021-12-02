#include "windows.h"
#include <stdio.h>
#include <cnn/cnn_lua.h>

#include "ui.h"
#include "Setup.h"
#include "conio2/conio2.h"



int cnnMain(int nargs, char **args) {
	printf("%d\n", nargs);
	for (int i = 0; i < nargs; ++i) {
		printf("%s\n", args[i]);
	}
	Setup s = Setup_new();
	GUI.can_run = &s->can_run;
	GUI.force_end = &s->force_end;
	char luaFile[250] = {0};
	getLuaFILE(luaFile, 250, nargs, (const char **) args);
	s->loadLua(s, luaFile);
	if (s->ok(s)) {
		s->runing = 1;
		Handle hload = Thread_new(s->loadImagens, s);
//		showCursor(0);
		int y = wherey() + 1;
		while (s->runing) {
//			gotoxy(1, y);
			GUI.setText(GUI.status, "Carregando Imagens: %d de %d        \n", s->iLoad.imAtual, s->iLoad.imTotal);
			Sleep(100);
		}
//		showCursor(1);
		Thread_Release(hload);
	}
	if (s->ok(s)) {
		s->runing = 1;
		HANDLE hload = Thread_new(s->loadLabels, s);
//		showCursor(0);
		int y = wherey() + 1;
		while (s->runing) {
//			gotoxy(1, y);
			GUI.setText(GUI.status, "Carregando labels: %d de %d        \n", s->iLoad.imAtual, s->iLoad.imTotal);
			Sleep(100);
		}
		showCursor(1);
		Thread_Release(hload);
	}

//	s->treinar(s);
	if (s->ok(s)) {
		s->runing = 1;
		Handle htreino = Thread_new(s->treinar, s);
		int y = wherey() + 1;
//		showCursor(0);
		GUI.setText(GUI.status, "Treinando");
		while (s->runing) {
			gotoxy(1, y);
			s->checkStop(s, "qQ");
//			GUI.setText(GUI.status,"Aperte 'q' para encerrar\n");
			GUI.setText(GUI.epoca, "%d de %d", s->itrain.epAtual, s->itrain.epTotal);
			GUI.setText(GUI.imagem, "%d de %d", s->itrain.imAtual, s->itrain.imTotal);
			GUI.setText(GUI.mse, "%.14lf", s->itrain.mse);
			GUI.setText(GUI.winHate, "%lf", s->itrain.winRate);
			GUI.setText(GUI.imps, "%lf", s->itrain.imps);
			Sleep(100);
		}
		showCursor(1);
		Thread_Release(htreino);
	}
	char *tmp = asprintf(NULL, "MSE %lf\nwin hate %lf.\n", s->itrain.mse, s->itrain.winRate);
	dialogBox("Treino terminado!", tmp);
	free_mem(tmp);
	return s->release(&s);
}
