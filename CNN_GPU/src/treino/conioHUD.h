#ifndef CONIO2_HUD_D
#define CONIO2_HUD_D
#ifndef MAX_STRING_LEN
#define MAX_STRING_LEN 256
#endif
#include "../conio2/conio2.h"
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <conio.h>

typedef struct {
	int *totalImagensTreino,
			*imagemAtual,
			*acertos,
			*epoca,
			*totalEpocas,
			*stop_listener;
	size_t *msInitEpoca, *msInitTrain;
	double *erro;

} InfoTrain;
typedef struct {
	char names[MAX_STRING_LEN];
} Nomes;
typedef struct {
	int nClasses;
	int *totalImagensTeste,
			*imagemAtual,
			*acertos,
			*stop_listener;
	size_t *msInitTest;
	int *estatisticasAcerto, *ncasos;
	Nomes *classesName;
	int dx;
} InfoTeste;

void hidecursor() {
	HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_CURSOR_INFO info;
	info.dwSize = 100;
	info.bVisible = FALSE;
	SetConsoleCursorInfo(consoleHandle, &info);
}

double getns() {
	struct timespec t = {0};
	clock_gettime(CLOCK_MONOTONIC, &t);
	double ns = (double) t.tv_sec * 1.0e9 + t.tv_nsec;
	return ns;
}

double getms() {
	return getns() * 1.0e-6;
}

void printBk(char *msg, int mx1, int color1, int color2) {
	int i;
	textbackground(color1);
	for (i = 0; msg[i] && i < mx1; i++) {
		printf("%c", msg[i]);
	}
	textbackground(color2);
	for (; msg[i]; i++) {
		printf("%c", msg[i]);
	}
}

void printTime(unsigned long long int t) {
	t = t/1000;
	printf("%02d:", (int) t / 3600);
	t = t % 3600;
	printf("%02d:", (int) t / 60);
	t = t % 60;
	printf("%02d", (int) t );

}

void *showInfoTrain(InfoTrain *info) {
	int y0 = 4;
	int perc;
	char buf[250] = {0};
	size_t imagemFeitas, imagensFaltando;
	size_t t,t2;

	clrscr();
	gotoxy(10, 2);
	printf("Para encerrar o treinamento aperte Q");
	gotoxy(10, y0);
	printf("Epoca ");
	gotoxy(10, y0 + 2);
	printf("Acertos");
	gotoxy(10, y0 + 1);
	printf("imagem ");
	gotoxy(44, y0 + 1);
	printf("Tempo restante ");
	gotoxy(44, y0);
	printf("Tempo restante ");
	gotoxy(10, y0 + 4);
	printf("Erro");
	gotoxy(28, y0 + 4);

	while (!*info->stop_listener) {
		imagemFeitas = *info->imagemAtual + 1;
		imagensFaltando = *info->totalImagensTreino - imagemFeitas;
		hidecursor();

		perc = (int) (*info->epoca / (double) *info->totalEpocas * 22.0);
		gotoxy(18, y0);
		snprintf(buf, 250, "% 3d de % 3d  %03.2f%%", *info->epoca, *info->totalEpocas,
		         *info->epoca / (double) *info->totalEpocas * 100.0);
		printBk(buf, perc, GREEN, BLACK);

		gotoxy(18, y0 + 2);
		printf("%3d de %3d  %03.2f%%", *info->acertos, *info->imagemAtual + 1,
		       *info->acertos / (*info->imagemAtual + 1.0) * 100.0);

		t = getms() - *info->msInitEpoca;
		t2 = getms() - *info->msInitTrain;
		gotoxy(60, y0 + 1);
		printf("-");
		printTime((size_t) (t / (double) imagemFeitas * imagensFaltando));

		gotoxy(18, y0 + 1);
		perc = (int) (*info->imagemAtual / (double) *info->totalImagensTreino * 22.0);
		snprintf(buf, 250, "% 3d de % 3d  %03.2f%%", *info->imagemAtual, *info->totalImagensTreino,
		         (*info->imagemAtual * 100.0) / *info->totalImagensTreino);
		printBk(buf, perc, GREEN, BLACK);


		imagemFeitas += *info->totalImagensTreino * *info->epoca;
		imagensFaltando += *info->totalImagensTreino * (*info->totalEpocas - *info->epoca);
		gotoxy(60, y0);
		printf("-");

		printTime((size_t) (t2 / (double) imagemFeitas * imagensFaltando) );

		gotoxy(18, y0 + 4);
		printf("%08.8lf", *info->erro);
		Sleep(200);
	}
	return NULL;
}

void *showInfoTest(InfoTeste *info) {
	int y0 = 4;
	int dx = info->dx;
	int perc;
	char buf[250] = {0};
	size_t imagemFeitas, imagensFaltando;
	size_t t;

	clrscr();
	gotoxy(10, 2);
	printf("Para encerrar o Teste aperte Q");
	gotoxy(10, y0);
	printf("Acerto total");

	gotoxy(10, y0 + 1);
	printf("imagem ");

	gotoxy(44, y0 + 1);
	printf("Tempo restante ");
	gotoxy(10, y0 + 4);
	printf("classes");
	gotoxy(10 + dx, y0 + 4);
	printf("acertos");
	gotoxy(10 + 2 * dx, y0 + 4);
	printf("Casos");

	for (int c = 0; c < info->nClasses; c++) {
		gotoxy(10, y0 + 5 + c);
		if (info->classesName != NULL)
			printf("%s", info->classesName[c].names);
		else
			printf("% 3d", c);
	}
	while (!*info->stop_listener) {
		imagemFeitas = *info->imagemAtual + 1;
		imagensFaltando = *info->totalImagensTeste - imagemFeitas;
		hidecursor();

		t = getms() - *info->msInitTest;
		gotoxy(60, y0 + 1);
		printf("-");
		printTime((size_t) (t / (double) imagemFeitas * imagensFaltando));

		gotoxy(22, y0);
		printf("% 3d de % 3d  % 3.2f%%", *info->acertos, *info->imagemAtual + 1,
		       *info->acertos / (*info->imagemAtual + 1.0) * 100.0);
		gotoxy(22, y0 + 1);
		perc = (int) (*info->imagemAtual / (double) *info->totalImagensTeste * 22.0);
		snprintf(buf, 250, "% 3d de % 3d  % 3.2f%%", *info->imagemAtual, *info->totalImagensTeste,
		         (*info->imagemAtual * 100.0) / *info->totalImagensTeste);
		printBk(buf, perc, GREEN, BLACK);

		for (int c = 0; c < info->nClasses; c++) {
			gotoxy(10 + dx, y0 + 5 + c);
			printf("% 4d", info->estatisticasAcerto[c]);
			gotoxy(10 + 2 * dx, y0 + 5 + c);
			printf("% 4d", info->ncasos[c]);
		}
		Sleep(200);
	}
	return NULL;
}

int testeConio2HUD() {
	hidecursor();
	int totalImagens = 3;
	int caso = 1;
	double erro = 13.5;
	double erros[3] = {1, 5, 6};
	int estatisticasAcerto[3] = {0}, ncasos[3] = {0};
	double estatiscaErro[3];
	Nomes classesName[3] = {(Nomes) {"zero"}, (Nomes) {"um"}, (Nomes) {"dois"}};
	int acertos = 1;
	size_t t0 = getms();
	char p;
	int stop = 0;
	InfoTeste info = {3, &totalImagens, &caso, &acertos, &stop, &t0, estatisticasAcerto, ncasos,
	                  classesName, 20};
	pthread_t tid;

	pthread_create(&tid, NULL, (void *(*)(void *)) showInfoTest, (void *) &info);

	getch();
	stop = 0;
	return 0;
}

#endif