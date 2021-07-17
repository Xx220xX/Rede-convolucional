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
#include <stdatomic.h>

typedef struct {
	int totalImagensTreino,
			imagemAtual,
			acertos,
			epoca,
			totalEpocas;
	size_t msInitEpoca, msInitTrain;
	double erro;
	atomic_int finish;
	int init;
} InfoTrain;
typedef struct {
	char names[MAX_STRING_LEN];
} Nomes;
typedef struct {
	int nClasses;
	int totalImagensTeste,
			imagemAtual,
			acertos;
	size_t msInitTest;
	int *estatisticasAcerto, *ncasos;
	Nomes *classesName;
	int dx;
	atomic_int finish;
	int init;
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
double getsec(){
	return getns()*1e-9;
}

void printBk(char *msg, double  mx1, int color1, int color2) {
	int i;
	textbackground(color1);
	int perc = mx1* strlen(msg);
	for (i = 0; msg[i] && i < perc; i++) {
		printf("%c", msg[i]);
	}
	textbackground(color2);
	for (; msg[i]; i++) {
		printf("%c", msg[i]);
	}
}

void printTime(unsigned long long int t) {
	t = t / 1000;
	printf("%02d:", (int) t / 3600);
	t = t % 3600;
	printf("%02d:", (int) t / 60);
	t = t % 60;
	printf("%02d", (int) t);

}

void *showInfoTrain(InfoTrain *info) {
	int y0 = 4;
	int perc;
	char buf[250] = {0};
	size_t imagemFeitas, imagensFaltando;
	size_t t, t2;
	double pc;
	if(!info->init) {
		hidecursor();
		clrscr();
		hidecursor();
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
		gotoxy(10, y0 + 5);
		printf("Imagens/s");
		gotoxy(28, y0 + 4);
		info->init = 1;
	}
	imagemFeitas = info->imagemAtual + 1;
	imagensFaltando = info->totalImagensTreino - imagemFeitas;
	gotoxy(18, y0);
	pc = (double)(info->epoca*info->totalImagensTreino+info->imagemAtual) / ((double) (info->totalEpocas*info->totalImagensTreino)) * 100.0;
	perc = (int) pc;
	pc = (pc - perc) * 100;
	snprintf(buf, 250, "% 6d de % 6d  % 3d.%02d%%", info->epoca+1, info->totalEpocas, perc, (int) pc);
	printBk(buf, (info->epoca*info->totalImagensTreino+info->imagemAtual) / ((double) info->totalEpocas*info->totalImagensTreino) , GREEN, BLACK);

	gotoxy(18, y0 + 2);
	pc = info->acertos / (info->imagemAtual + 1.0) * 100.0;
	perc = (int) pc;
	pc = (pc - perc) * 100;
	printf("%6d de %6d  % 3d.%02d%%", info->acertos, info->imagemAtual + 1, perc, (int) pc);
	t = getms() - info->msInitEpoca;
	t2 = getms() - info->msInitTrain;

	gotoxy(60, y0 + 1);
	printf("-");
	printTime((size_t) (t / (double) imagemFeitas * imagensFaltando));
	gotoxy(18, y0 + 1);
	pc = (info->imagemAtual * 100.0) / info->totalImagensTreino;
	perc = (int) pc;
	pc = (pc - perc) * 100;
	snprintf(buf, 250, "% 6d de % 6d  % 3d.%02d%%", info->imagemAtual, info->totalImagensTreino,
	         perc, (int) pc);
	printBk(buf, info->imagemAtual / (double) info->totalImagensTreino, GREEN, BLACK);
	imagemFeitas += info->totalImagensTreino * info->epoca;
	imagensFaltando += info->totalImagensTreino * (info->totalEpocas - info->epoca);
	gotoxy(60, y0);
	printf("-");
	printTime((size_t) (t2 / (double) imagemFeitas * imagensFaltando));
	gotoxy(20, y0 + 4);
	pc =info->erro;
	perc = (int) pc;
	pc = (pc - perc) * 1e8;
	printf("%3d.%08d",perc,(int)pc);


	pc = (info->imagemAtual+1.0)/(getsec() - info->msInitEpoca*1e-3);
	perc = (int)pc;
	gotoxy(20, y0 + 5);
	printf("%8d",perc);
	Sleep(200);
	info->finish = 1;
	return NULL;
}

void *showInfoTest(InfoTeste *info) {
	int y0 = 4;
	int dx = info->dx+10;
	int perc;
	double pc;
	char buf[250] = {0};
	size_t imagemFeitas, imagensFaltando;
	size_t t;
	if(!info->init) {
		clrscr();
		hidecursor();
		gotoxy(10, 2);
		printf("Para encerrar o Teste aperte Q");
		gotoxy(10, y0);
		printf("Acerto total");

		gotoxy(10, y0 + 1);
		printf("imagem ");

		gotoxy(44, y0 + 1);
		printf("Tempo restante ");
		gotoxy(10, y0 + 5);
		printf("classes");
		gotoxy(10 + dx, y0 + 5);
		printf("acertos");
		gotoxy(10 + 2 * dx, y0 + 5);
		printf("Casos");

		for (int c = 0; c < info->nClasses; c++) {
			gotoxy(10, y0 + 6 + c);
			if (info->classesName != NULL)
				printf("%s", info->classesName[c].names);
			else
				printf("% 3d", c);
		}
		Sleep(10);
		hidecursor();
		info->init = 1;
	}
	imagemFeitas = info->imagemAtual + 1;
	imagensFaltando = info->totalImagensTeste - imagemFeitas;


	t = getms() - info->msInitTest;
	gotoxy(60, y0 + 1);
	printf("-");
	printTime((size_t) (t / (double) imagemFeitas * imagensFaltando));

	gotoxy(22, y0);
	pc = info->acertos / (info->imagemAtual + 1.0) * 100.0;
	perc = (int) pc;
	pc = (pc - perc) * 100;
	printf("% 3d de % 3d  % 3d.%02d%%", info->acertos, info->imagemAtual + 1, perc, (int) pc);
	gotoxy(22, y0 + 1);
	pc = (info->imagemAtual * 100.0) / info->totalImagensTeste;
	perc = (int) pc;
	pc = (pc - perc) * 100;
	snprintf(buf, 250, "% 3d de % 3d  % 3d.%2d%%", info->imagemAtual, info->totalImagensTeste,
	         perc, (int) pc);
	printBk(buf, info->imagemAtual / (double) info->totalImagensTeste , GREEN, BLACK);

	for (int c = 0; c < info->nClasses; c++) {
		gotoxy(10 + dx, y0 + 7 + c);
		printf("% 4d", info->estatisticasAcerto[c]);
		gotoxy(10 + 2 * dx, y0 + 7 + c);
		printf("% 4d", info->ncasos[c]);
	}

	Sleep(200);
	info->finish = 1;
	return NULL;
}


#endif