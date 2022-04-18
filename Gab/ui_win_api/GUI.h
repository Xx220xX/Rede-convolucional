//
// Created by Henrique on 29/12/2021.
//

#ifndef UI_GUI_H
#define UI_GUI_H

#include "plot.h"

#define  USE_PROGRESS_BAR  0
#define GUI_UPDATE_WINDOW 14321853185

#include "setup/Setup.h"

typedef HWND Button;
typedef HWND TextLabel;
typedef HWND ProgressBar[2];

#define IDM_FITNES 0x12

struct {
	TextLabel status;
	TextLabel *labels;
	int nlabels;
	ProgressBar *progress;
	int nProgress;
	atomic_int *can_run;
	atomic_int *force_end;
	atomic_int endDraw;
	HWND hmain;
//	int w;
//	int h;
	void *arg;

	Figure figs[10];
	int nfigs;

	void (*setText)(TextLabel text, const char *format, ...);

	void (*setProgress)(ProgressBar progressBar, double value);

	void (*clearWindow)();

	void (*addLabel)(char *text, int x, int y, int w, int h);

	void (*make_loadImages)();

	void (*make_loadLabels)();

	void (*make_train)();

	void (*make_teste)();

	void (*updateLoadImagens)(int im, int total, double t0);

	void (*updateTrain)(double faval, Itrain itrain, double deltaT);

	void (*updateTeste)(int im, int total, double mse, double winhate, double deltaT);

	void (*draw)();

	void (*capture)(char *fileName);

	HINSTANCE hisntance;
	atomic_int avaliar;
	atomic_int avaliando;
	HMENU menu_fitnes_option;
	HMENU menu;
} GUI = {0};


void GUI_addLabel(char *text, int x, int y, int w, int h) {
	GUI.nlabels++;
	GUI.labels = realloc(GUI.labels, GUI.nlabels * sizeof(TextLabel));
	GUI.labels[GUI.nlabels - 1] = CreateWindowA("static", text, WS_CHILD | WS_VISIBLE, x, y, w, h, GUI.hmain, NULL, NULL, NULL);
}

Figure *GUI_addFig();

void GUI_releasefigs();

void GUI_clearWindow() {
	for (int i = 0; i < GUI.nlabels; ++i) {
		DestroyWindow(GUI.labels[i]);
	}
	if (GUI.labels) {
		free(GUI.labels);
	}
	GUI_releasefigs();
	GUI.nlabels = 0;
	GUI.labels = NULL;
	GUI.draw = NULL;
}

void GUI_loadImage() {
	GUI.clearWindow();
	GUI.addLabel("Progresso :", 1, 20, 100, 20);
	GUI.addLabel("", 100, 20, 100, 20);
	GUI.addLabel("Tempo restante estimado:", 1, 40, 200, 20);
	GUI.addLabel("", 200, 40, 100, 20);
}

int CaptureAnImage(char *fileName, HWND hWnd);

void GUI_capture(char *filename) {
	while (!GUI.endDraw);
	GUI.arg = filename;
	CaptureAnImage(filename, GUI.hmain);
//	GUI.draw = (void (*)()) CaptureAnImage;
//	RedrawWindow(GUI.hmain, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
}

void GUI_updateLoadImagens(int im, int total, double delta) {
	double progresso = im * 100.0 / total;
	double imps = im / delta;
	imps = (total - im) / imps;
	GUI.setText(GUI.labels[1], "%.2lf%%", progresso);
	GUI.setText(GUI.labels[3], "%.1lf s", imps);
}

void time2str(char *buf, int len, double dtm) {
	uint64_t tm = (uint64_t) abs(dtm);
	uint32_t dia, hora, minuto, segundo;
	dia = tm / 86400;
	tm %= 86400;
	hora = tm / 3600;
	tm %= 3600;
	minuto = tm / 60;
	segundo = tm % 60;
	int ln = 0;
	memset(buf, 0, len);
	if (dia) {
		ln += snprintf(buf + ln, len - ln, "%d dias ", dia);
	}
	if (hora) {
		ln += snprintf(buf + ln, len - ln, "%d horas ", hora);
	}
	if (minuto) {
		ln += snprintf(buf + ln, len - ln, "%d minutos ", minuto);
	}
	if (segundo > 1) {
		snprintf(buf + ln, len - ln, "%d segundos", segundo);
	} else {
		snprintf(buf + ln, len - ln, "%d segundo", segundo);
	}
}

static char *TEXT_winrate = "Win rate";
static char *TEXT_MSE = "Erro";
#define FIG_TOP 0
#define FIG_BOTTOM 0
#define  AXE_ERRO_T 0
#define  AXE_ERRO_A 1
#define  AXE_WIN_T 0
#define  AXE_WIN_A 3

void GUI_train() {
	int i = 1;
	int dy = 20;
	int w = 200;
	GUI.clearWindow();


	//	AppendMenuA(menu,MF_STRING,IDM_FITNES,"&")

	GUI.addLabel("Progresso treino:", 1, dy * i, w, dy);              //0
	GUI.addLabel("", w, dy * i++, w, dy);                           //1
	GUI.addLabel("Progresso epoca:", 1, dy * i, w, dy);               //2
	GUI.addLabel("", w, dy * i++, w, dy);                           //3
	GUI.addLabel("Tempo restante:", 1, dy * i, w, dy);       //4
	GUI.addLabel("", w, dy * i++, w, dy);                           //5
	GUI.addLabel("Imagens por segundo", 1, dy * i, w, dy);            //6
	GUI.addLabel("", w, dy * i, w, dy);                             //7
	GUI.addLabel("Erro", 1, dy * i, w, dy);                            //8
	GUI.addLabel("", w, dy * i++, w, dy);                             //9
	GUI.addLabel("Acertos", 1, dy * i, w, dy);                        //10
	GUI.addLabel("", w, dy * i++, w, dy);                                //11
	GUI.addLabel("imagens por segundo", 1, dy * i, w, dy);                      //12
	GUI.addLabel("", w, dy * i++, w, dy);                             //13

	Figure *f = GUI_addFig();
	Figure_new(f, GUI.hmain, GUI.hisntance, 10, dy * i++, 700, 250);
	f->bkcolor = RGB(0xff, 0xff, 0xff);
	f->grid = 1;
	f->nstepx = 10;
	f->nstepy = 0;
	f->ystep = 0.25;
	f->ymax = 1.2;
	f->ymin = -0.01;
	f->title = TEXT_MSE;
	f->xmin = 0;
	f->xmax = 1;
	f->ygrid_start = 0;
	f->xgrid_start = 0;
	f->wpad = 50;
	f->legend = 1;
	f->putAxe(f, RGB(0xff, 0, 0), "Treino");
	f->putAxe(f, RGB(0, 0xff, 0), "Avaliacao");
	f = GUI_addFig();
	Figure_new(f, GUI.hmain, GUI.hisntance, 10, dy * i + 250, 700, 250);
	f->bkcolor = RGB(0xff, 0xff, 0xff);
	f->grid = 1;
	f->legend = 1;

	f->nstepx = 10;
	f->ystep = 10;
	f->ygrid_start = 0;
	f->xgrid_start = 0;
	f->ymax = 100;
	f->ymin = -0.5;
	f->title = TEXT_winrate;
	f->xmin = 0;
	f->xmax = 1;
	f->wpad = 50;
	f->putAxe(f, RGB(0xff, 0, 0), "Estimado");
	f->putAxe(f, RGB(0, 0xff, 0), "Medio na Epoca");
	f->putAxe(f, RGB(0, 0, 0xff), "Medio");
	f->putAxe(f, RGB(0xff, 0, 0xff), "Avaliacao");

}

void GUI_updateTrain(double faval, Itrain itrain, double deltat) {
	if (faval > 0) {
		GUI.setText(GUI.labels[3], "avaliando %.2lf", faval);
		return;
	}
	if (itrain.epAtual == 0 || itrain.imAtual == 0) {
		return;
	}
	if (GUI.avaliando) {
		return;
	}
	if (deltat == 0) {
		return;
	}

	double progresso = itrain.imAtual * 100.0 / itrain.imTotal;
	int nimages = (itrain.epAtual - 1) * itrain.imTotal + itrain.imAtual;
	char tempo_str[250];
	double imps = itrain.imagensCalculadas / deltat;
	double tempo = (itrain.totalImages - itrain.imagensCalculadas) / imps;
	float epoca = nimages / (float) itrain.imTotal;
	static float lastEpoca = -10;
	if (epoca == lastEpoca) {
		return;
	}
	lastEpoca = epoca;
	time2str(tempo_str, 250, tempo);
	GUI.setText(GUI.labels[3], "%.2lf%% %d/%d", progresso, itrain.imAtual, itrain.imTotal);
	progresso = 100.0 * itrain.imagensCalculadas / (itrain.totalImages);
	GUI.setText(GUI.labels[1], "%.2lf%% %d/%d", progresso, itrain.epAtual, itrain.epTotal);
	GUI.setText(GUI.labels[5], "%s", tempo_str);
	GUI.setText(GUI.labels[9], "%lf", itrain.mse);
	GUI.setText(GUI.labels[11], "%lf%%", itrain.winRate);
	GUI.setText(GUI.labels[13], "%.1lf", imps);

	if (GUI.endDraw) {
		if (GUI.figs[0].xmax != itrain.epTotal) {
			GUI.figs[0].xmax = GUI.figs[1].xmax = itrain.epTotal;
//			GUI.figs[0].xmax = GUI.figs[1].xmax = ep;
			GUI.figs[0].draw(GUI.figs, NULL);
			GUI.figs[1].draw(GUI.figs + 1, NULL);
		}

		GUI.figs[0].axes[0].pushDraw(GUI.figs[0].axes, epoca, itrain.mse);
		GUI.figs[1].axes[0].pushDraw(GUI.figs[1].axes, epoca, itrain.winRate);
		GUI.figs[1].axes[1].pushDraw(GUI.figs[1].axes + 1, epoca, itrain.winRateMedio);
		GUI.figs[1].axes[2].pushDraw(GUI.figs[1].axes + 2, epoca, itrain.winRateMedioep);
	}
//	appendPoint(&GUI.graphico, nimages, winhate, 0, total * eptotal, 0, 100);
//	RedrawWindow(GUI.hmain, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);

}

void GUI_teste() {
	int i = 1;
	int dy = 20;
	int w = 200;
	GUI.clearWindow();
	GUI.addLabel(u8"Progresso Avaliação:", 1, dy * i, w, dy);         //0
	GUI.addLabel("", w, dy * i++, w, dy);                           //1
	GUI.addLabel("Tempo restante:", 1, dy * i, w, dy);             //2
	GUI.addLabel("", w, dy * i++, w, dy);                           //3
	GUI.addLabel("Mse", 1, dy * i, w, dy);                            //4
	GUI.addLabel("", w, dy * i++, w, dy);                             //5
	GUI.addLabel("Acertos", 1, dy * i, w, dy);                        //6
	GUI.addLabel("", w, dy * i++, w, dy);                                //7
	GUI.addLabel("imagens por segundo", 1, dy * i, w, dy);             //8
	GUI.addLabel("", w, dy * i++, w, dy);                             //9
	Figure *f = GUI_addFig();
	Figure_new(f, GUI.hmain, GUI.hisntance, 10, dy * i++, 500, 250);
	f->bkcolor = RGB(0xff, 0xff, 0xff);
	f->grid = 1;
	f->nstepx = 10;
	f->nstepy = 10;
	f->ymax = 1.05;
	f->ymin = -0.01;
	f->title = TEXT_MSE;
	f->xmin = 0;
	f->xmax = 1;
	f->ygrid_start = 0;
	f->xgrid_start = 0;
	f->wpad = 50;
	f->putAxe(f, RGB(0xff, 0, 0), "Erro");
	f = GUI_addFig();
	Figure_new(f, GUI.hmain, GUI.hisntance, 10, dy * i + 250, 500, 250);
	f->bkcolor = RGB(0xff, 0xff, 0xff);
	f->grid = 1;
	f->nstepx = 10;
	f->ystep = 10;
	f->ygrid_start = 0;
	f->xgrid_start = 0;
	f->ymax = 100;
	f->ymin = -0.5;
	f->title = TEXT_winrate;
	f->xmin = 0;
	f->xmax = 1;
	f->wpad = 50;
	f->putAxe(f, RGB(0xff, 0, 0), "Acerto");

//	GUI.graphico.x = 100;
//	GUI.graphico.y = dy * i++;
//	GUI.graphico.w = 500;
//	GUI.graphico.h = 300;
//	GUI.graphico.npoint = 0;
//	if (GUI.graphico.points) { free(GUI.graphico.points); }
//	GUI.graphico.points = NULL;
//	GUI.graphico.npoint = 0;
//	GUI.draw = GUI_draw;

}

void GUI_updateTeste(int im, int total, double mse, double winRate, double deltat) {
	if (im == 0) {
		return;
	}
	double progresso = im * 100.0 / total;
	char tempo_str[250];
	double imps = im / deltat;
	double tempo = (total - im) / imps;
	time2str(tempo_str, 250, tempo);
	GUI.setText(GUI.labels[1], "%.2lf%% %d/%d", progresso, im, total);
	progresso = 100.0 * im / (total);
	GUI.setText(GUI.labels[3], "%s", tempo_str);
	GUI.setText(GUI.labels[5], "%lf", mse);
	GUI.setText(GUI.labels[7], "%lf%%", winRate);
	GUI.setText(GUI.labels[9], "%.1lf", imps);
	if (GUI.endDraw) {
		if (GUI.figs[0].xmax != total) {
			GUI.figs[0].xmax = GUI.figs[1].xmax = total;
			GUI.figs[0].draw(GUI.figs, NULL);
			GUI.figs[1].draw(GUI.figs + 1, NULL);
		}
		GUI.figs[0].axes[0].pushDraw(GUI.figs[0].axes, im, mse);
		GUI.figs[1].axes[0].pushDraw(GUI.figs[1].axes, im, winRate);
	}

//	appendPoint(&GUI.graphico, im, winhate, 0, total, 0, 100);
//	RedrawWindow(GUI.hmain, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);

}


void GUI_make_loadImages() {
	GUI.setText(GUI.status, "Carregando imagems");
	GUI.endDraw = 0;
	PostMessageA(GUI.hmain, WM_UPDATEUISTATE, (WPARAM) GUI_loadImage, GUI_UPDATE_WINDOW);
}

void GUI_make_loadLabels() {
	GUI.setText(GUI.status, "Carregando Labels");
	GUI.endDraw = 0;
	PostMessageA(GUI.hmain, WM_UPDATEUISTATE, (WPARAM) GUI_loadImage, GUI_UPDATE_WINDOW);
}

void GUI_make_train() {
	GUI.setText(GUI.status, "Treinando");
	GUI.endDraw = 0;
	PostMessageA(GUI.hmain, WM_UPDATEUISTATE, (WPARAM) GUI_train, GUI_UPDATE_WINDOW);
}

void GUI_make_teste() {
	GUI.setText(GUI.status, "Avaliando");
	GUI.endDraw = 0;
	PostMessageA(GUI.hmain, WM_UPDATEUISTATE, (WPARAM) GUI_teste, GUI_UPDATE_WINDOW);
}


void GUI_setProgress(ProgressBar progressBar, double value) {
	int ivalue = value;
#if (USE_PROGRESS_BAR == 1)
	SendMessage(progressBar[0], PBM_SETPOS, (WPARAM) ivalue % 101, 0);
	GUI_setText(progressBar[1], "%.2lf%%", value);
#endif
}

void GUI_setText(TextLabel text, const char *format, ...) {
	char *msg = NULL;
	va_list v;
	va_start(v, format);
	size_t len = vsnprintf(NULL, 0, format, v) + 1;
	msg = calloc(len, 1);
	vsnprintf(msg, len, format, v);
	SetWindowTextA(text, msg);
	free(msg);
	va_end(v);
}

int wstrlen(const LPWSTR lpwstr) {
	int len = 0;
	for (; lpwstr[len]; ++len) {
	}
	return len;
}

void GUI_init(HWND hwnd) {
	GUI.setText = GUI_setText;
	GUI.setProgress = GUI_setProgress;
	GUI.make_loadImages = GUI_make_loadImages;
	GUI.make_loadLabels = GUI_make_loadLabels;
	GUI.clearWindow = GUI_clearWindow;
	GUI.addLabel = GUI_addLabel;
	GUI.updateLoadImagens = GUI_updateLoadImagens;
	GUI.make_train = GUI_make_train;
	GUI.updateTrain = GUI_updateTrain;
	GUI.make_teste = GUI_make_teste;
	GUI.updateTeste = GUI_updateTeste;
	GUI.capture = GUI_capture;
	GUI.hmain = hwnd;
}


Figure *GUI_addFig() {
//	GUI.figs = realloc(GUI.figs, (GUI.nfigs + 1) * sizeof(Figure));
	GUI.figs[GUI.nfigs] = (Figure) {0};
	GUI.nfigs++;
	return &GUI.figs[GUI.nfigs - 1];
}

void GUI_releasefigs() {
	for (int i = 0; i < GUI.nfigs; ++i) {
		GUI.figs[i].release(GUI.figs + i);
	}
	GUI.nfigs = 0;

}

#endif //UI_GUI_H
