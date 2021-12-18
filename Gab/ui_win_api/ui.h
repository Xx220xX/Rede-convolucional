//
// Created by Henrique on 01/12/2021.
//

#ifndef GAB_UI_H
#define GAB_UI_H

#include "windows.h"
#include "windowsx.h"
#include "error_list.h"
#include "thread/Thread.h"
#include <wchar.h>
#include <strsafe.h>

#include <commctrl.h>
#include <locale.h>
//#include "client.h"

#define cnnMain _local_main_
#define  USE_PROGRESS_BAR  0
#define  GUI_UPDATE_WINDOW 120132219

int cnnMain(int, char **);

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

void CreateLabels(HWND);

void checkArgs();

typedef HWND TextLabel;
typedef HWND ProgressBar[2];
typedef struct {
	int instacied;
	int x, y;
	int w, h;
	POINT *points;
	int npoint;
	DWORD color;
} Graph_area;

struct {
	TextLabel status;
	TextLabel *labels;
	int nlabels;
	ProgressBar *progress;
	int nProgress;
	atomic_int *can_run;
	atomic_int *force_end;
	HWND hmain;
	HWND hconsole;
	Graph_area graphico;
	void *arg;

	void (*setText)(TextLabel text, const char *format, ...);

	void (*setProgress)(ProgressBar progressBar, double value);

	void (*clearWindow)();

	void (*addLabel)(char *text, int x, int y, int w, int h);

	void (*make_loadImages)();

	void (*make_loadLabels)();

	void (*make_train)();

	void (*make_teste)();

	void (*updateLoadImagens)(int im, int total, double t0);

	void (*updateTrain)(int im, int total, int ep, int eptotal, double mse, double winhate, double deltaT);

	void (*updateTeste)(int im, int total, double mse, double winhate, double deltaT);

	void (*draw)();

	void (*capture)(char *fileName);
} GUI = {0};

void GUI_addLabel(char *text, int x, int y, int w, int h) {
	GUI.nlabels++;
	GUI.labels = realloc(GUI.labels, GUI.nlabels * sizeof(TextLabel));
	GUI.labels[GUI.nlabels - 1] = CreateWindowA("static", text, WS_CHILD | WS_VISIBLE, x, y, w, h, GUI.hmain, NULL, NULL, NULL);
}

void GUI_clearWindow() {
	for (int i = 0; i < GUI.nlabels; ++i) {
		DestroyWindow(GUI.labels[i]);
	}
	if (GUI.labels) {
		free(GUI.labels);
	}
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

int CaptureAnImage();

void GUI_capture(char *filename) {
	GUI.arg = filename;
	GUI.draw = (void (*)()) CaptureAnImage;
	RedrawWindow(GUI.hmain, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);


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

void GUI_draw() {
	HDC hdc;
	PAINTSTRUCT ps;
	hdc = BeginPaint(GUI.hmain, &ps);
//	HBRUSH hBrush1 = CreateSolidBrush(RGB(255, 0, 255));
//	HBRUSH holdBrush = SelectObject(hdc, hBrush1);
//	Rectangle(hdc, GUI.treino.x, GUI.treino.y, GUI.treino.w, GUI.treino.h);
//	DeleteObject(hBrush1);
//	for (int i = 0; i < GUI.treino.npoint; ++i) {
//		printf("%ld %ld\n", GUI.treino.points[i].x, GUI.treino.points[i].y);
//	}
	POINT esquerdo[] = {GUI.graphico.x, GUI.graphico.y, GUI.graphico.x, GUI.graphico.y + GUI.graphico.h};
	POINT direito[] = {GUI.graphico.x + GUI.graphico.w, GUI.graphico.y, GUI.graphico.x + GUI.graphico.w, GUI.graphico.y + GUI.graphico.h};
	POINT cima[] = {GUI.graphico.x, GUI.graphico.y, GUI.graphico.x + GUI.graphico.w, GUI.graphico.y};
	POINT baixo[] = {GUI.graphico.x, GUI.graphico.y + GUI.graphico.h, GUI.graphico.x + GUI.graphico.w, GUI.graphico.y + GUI.graphico.h};
	Polyline(hdc, esquerdo, 2);
	Polyline(hdc, direito, 2);
	Polyline(hdc, cima, 2);
	Polyline(hdc, baixo, 2);
	Polyline(hdc, GUI.graphico.points, GUI.graphico.npoint);
	EndPaint(GUI.hmain, &ps);
}

void GUI_train() {
	int i = 1;
	int dy = 20;
	int w = 200;
	GUI.clearWindow();
	GUI.addLabel("Progresso treino:", 1, dy * i, w, dy);              //0
	GUI.addLabel("", w, dy * i++, w, dy);                           //1
	GUI.addLabel("Progresso epoca:", 1, dy * i, w, dy);               //2
	GUI.addLabel("", w, dy * i++, w, dy);                           //3
	GUI.addLabel("Tempo restante:", 1, dy * i, w, dy);       //4
	GUI.addLabel("", w, dy * i++, w, dy);                           //5
	GUI.addLabel("Imagens por segundo", 1, dy * i, w, dy);            //6
	GUI.addLabel("", w, dy * i, w, dy);                             //7
	GUI.addLabel("Mse", 1, dy * i, w, dy);                            //8
	GUI.addLabel("", w, dy * i++, w, dy);                             //9
	GUI.addLabel("Acertos", 1, dy * i, w, dy);                        //10
	GUI.addLabel("", w, dy * i++, w, dy);                                //11
	GUI.addLabel("imagens por segundo", 1, dy * i, w, dy);                      //12
	GUI.addLabel("", w, dy * i++, w, dy);                             //13

	GUI.graphico.x = 100;
	GUI.graphico.y = dy * i++;
	GUI.graphico.w = 500;
	GUI.graphico.h = 300;
	GUI.graphico.npoint = 0;
	if (GUI.graphico.points) { free(GUI.graphico.points); }
	GUI.graphico.points = NULL;
	GUI.graphico.npoint = 0;
	GUI.draw = GUI_draw;
}

void appendPoint(Graph_area *ga, double x, double y, double lx, double hx, double ly, double hy) {
	ga->npoint++;
	ga->points = realloc(ga->points, ga->npoint * sizeof(POINT));
	x = ga->x + x / (hx - ly) * ga->w;
	y = ga->y + ga->h - y / (hy - ly) * ga->h;
	ga->points[ga->npoint - 1].x = x;
	ga->points[ga->npoint - 1].y = y;
}


void GUI_updateTrain(int im, int total, int ep, int eptotal, double mse, double winhate, double deltat) {
	double progresso = im * 100.0 / total;
	int nimages = (ep - 1) * total + im;
	char tempo_str[250];
	double imps = nimages / deltat;
	double tempo = (total * eptotal - nimages) / imps;

	time2str(tempo_str, 250, tempo);
	GUI.setText(GUI.labels[3], "%.2lf%% %d/%d", progresso, im, total);
	progresso = 100.0 * nimages / (eptotal * total);
	GUI.setText(GUI.labels[1], "%.2lf%% %d/%d", progresso, ep, eptotal);
	GUI.setText(GUI.labels[5], "%s", tempo_str);
	GUI.setText(GUI.labels[9], "%lf", mse);
	GUI.setText(GUI.labels[11], "%lf%%", winhate);
	GUI.setText(GUI.labels[13], "%.1lf", imps);

	appendPoint(&GUI.graphico, nimages, winhate, 0, total * eptotal, 0, 100);
	RedrawWindow(GUI.hmain, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);

}

void GUI_teste() {
	int i = 1;
	int dy = 20;
	int w = 200;
	GUI.clearWindow();
	GUI.addLabel("Progresso Avaliação:", 1, dy * i, w, dy);         //0
	GUI.addLabel("", w, dy * i++, w, dy);                           //1
	GUI.addLabel("Tempo restante:", 1, dy * i, w, dy);             //2
	GUI.addLabel("", w, dy * i++, w, dy);                           //3
	GUI.addLabel("Mse", 1, dy * i, w, dy);                            //4
	GUI.addLabel("", w, dy * i++, w, dy);                             //5
	GUI.addLabel("Acertos", 1, dy * i, w, dy);                        //6
	GUI.addLabel("", w, dy * i++, w, dy);                                //7
	GUI.addLabel("imagens por segundo", 1, dy * i, w, dy);             //8
	GUI.addLabel("", w, dy * i++, w, dy);                             //9
	GUI.graphico.x = 100;
	GUI.graphico.y = dy * i++;
	GUI.graphico.w = 500;
	GUI.graphico.h = 300;
	GUI.graphico.npoint = 0;
	if (GUI.graphico.points) { free(GUI.graphico.points); }
	GUI.graphico.points = NULL;
	GUI.graphico.npoint = 0;
	GUI.draw = GUI_draw;

}

void GUI_updateTeste(int im, int total, double mse, double winhate, double deltat) {
	double progresso = im * 100.0 / total;
	char tempo_str[250];
	double imps = im / deltat;
	double tempo = (total - im) / imps;
	time2str(tempo_str, 250, tempo);
	GUI.setText(GUI.labels[1], "%.2lf%% %d/%d", progresso, im, total);
	progresso = 100.0 * im / (total);
	GUI.setText(GUI.labels[3], "%s", tempo_str);
	GUI.setText(GUI.labels[5], "%lf", mse);
	GUI.setText(GUI.labels[7], "%lf%%", winhate);
	GUI.setText(GUI.labels[9], "%.1lf", imps);

	appendPoint(&GUI.graphico, im, winhate, 0, total, 0, 100);
	RedrawWindow(GUI.hmain, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);

}


void GUI_make_loadImages() {
	GUI.setText(GUI.status, "Carregando imagems");
	PostMessageA(GUI.hmain, WM_UPDATEUISTATE, (WPARAM) GUI_loadImage, GUI_UPDATE_WINDOW);
}

void GUI_make_loadLabels() {
	GUI.setText(GUI.status, "Carregando Labels");
	PostMessageA(GUI.hmain, WM_UPDATEUISTATE, (WPARAM) GUI_loadImage, GUI_UPDATE_WINDOW);
}

void GUI_make_train() {
	GUI.setText(GUI.status, "Treinando");
	PostMessageA(GUI.hmain, WM_UPDATEUISTATE, (WPARAM) GUI_train, GUI_UPDATE_WINDOW);
}

void GUI_make_teste() {
	GUI.setText(GUI.status, "Avaliando");
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
	for (; lpwstr[len]; ++len) {}
	return len;
}

void run_main(PSTR pCmdLine) {
//	Client_connect();
	cnnMain(__argc, __argv);
//	Client_close();
	PostMessageA(GUI.hmain, WM_DESTROY, 0, 0);

}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int CmdShow) {
	checkArgs();
//	GdiplusStartupInput gdii;
//	ULONG_PTR gditoken;
//	GdiplusStartup(&gditoken,&gdii,NULL);

	GUI.hconsole = FindWindowA("ConsoleWindowClass", NULL);
	HWND hwnd;
	MSG msg;

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


	WNDCLASSW wc = {0};
	wc.lpszClassName = L"GabCnn";
	wc.hInstance = hInstance;
	wc.hbrBackground = GetSysColorBrush(COLOR_3DFACE);
	wc.lpfnWndProc = WndProc;
	wc.hCursor = LoadCursor(0, IDC_ARROW);
	RegisterClassW(&wc);
	hwnd = CreateWindowW(wc.lpszClassName, L"GabCnn", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 150, 150, 750, 500, GUI.hconsole, 0, hInstance, 0);

	GUI.hmain = hwnd;
	SetWindowTextA(hwnd, "Gab Cnn");
	if (!hwnd) {
		fprintf(stderr, "Falha ao criar janela\n");
		exit(-1);
	}
	Handle hmain = Thread_new(run_main, pCmdLine);
	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
//			RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);

	}
	Thread_Release(hmain);
//	GdiplusShutdown(gditoken);
	return (int) msg.wParam;
}

void checkArgs() {
	int nargs = __argc;
	char **args = __argv;
	system("chcp 65001 | echo off");
	for (int i = 1; i < nargs; ++i) {
		if (!strcmp(args[i], "--v") || !strcmp(args[i], "--version")) {
			printf("Gabriela IA\n");
			printf("versão %s\n", Cnn_version());
			printf("https://xx220xx.github.io/Rede-convolucional/\n\n");
			exit(0);
		}
	}
	if (nargs > 2) {
		fprintf(stderr, "Argumentos invalidos\n");
		exit(GAB_INVALID_PARAM);
	}
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
//	show_image(hwnd);
//	Sleep(10);
	switch (msg) {
		case WM_CREATE:
			GUI.hmain = hwnd;
			GUI.status = CreateWindowA("static", "Gabriela iniciada", WS_CHILD | WS_VISIBLE, 1, 1, 200, 20, GUI.hmain, NULL, NULL, NULL);
			GUI_addLabel("Pronta para aprender", 1, 20, 200, 20);
//			CreateLabels(hwnd);
			break;
		case WM_UPDATEUISTATE:
			if (lParam == GUI_UPDATE_WINDOW) {
				((void (*)()) wParam)();
			}
			break;
		case WM_MOVE:
			break;
		case WM_CLOSE:
			if (MessageBoxA(hwnd, "Really quit?", "Gab", MB_OKCANCEL) == IDOK) {
				if (GUI.can_run && GUI.force_end) {
					*GUI.can_run = 0;
					*GUI.force_end = 1;
				}
				break;
			}
			return 0;// User canceled. Do nothing.
		case WM_PAINT:
			if (GUI.draw) {
				GUI.draw();
			}
			break;

		case WM_DESTROY:
			PostQuitMessage(0);
			break;
	}

	return DefWindowProcW(hwnd, msg, wParam, lParam);
}

void CreateLabels(HWND hwnd) {
/*	int x = 1, h = 25, w = 150, dy = 0;
	int dh = 25;
	int y = 10;
#if (USE_PROGRESS_BAR == 1)
	GUI.progress[1] = CreateWindowA("static", "status: ",
									WS_CHILD | WS_VISIBLE|BS_TEXT,
									x +  3.1*w, y + (dy) * dh, w, h,
									hwnd, NULL, NULL, NULL);
	GUI.progress[0] = CreateWindowA(PROGRESS_CLASS, (LPTSTR) NULL,
									WS_CHILD | WS_VISIBLE ,
									x + w, y + (dy++) * dh, 2 * w, h,
									hwnd, NULL, NULL, NULL);
#endif

//	HWND Stealth;
//	AllocConsole();
//	ShowWindow(GUI.hconsole,0);
	CreateWindowA("static", "status: ", WS_CHILD | WS_VISIBLE, x, y + (dy) * dh, w, h, hwnd, NULL, NULL, NULL);

	GUI.status = CreateWindowA("static", "", WS_CHILD | WS_VISIBLE, x + w, y + (dy++) * dh, 1000, h, hwnd, NULL, NULL, NULL);

	CreateWindowA("static", "epoca: ", WS_CHILD | WS_VISIBLE, x, y + (dy) * dh, w, h, hwnd, NULL, NULL, NULL);
	GUI.epoca = CreateWindowA("static", "", WS_CHILD | WS_VISIBLE, x + w, y + (dy++) * dh, w, h, hwnd, NULL, NULL, NULL);

	CreateWindowA("static", "imagem: ", WS_CHILD | WS_VISIBLE, x, y + (dy) * dh, w, h, hwnd, NULL, NULL, NULL);

	GUI.imagem = CreateWindowA("static", "", WS_CHILD | WS_VISIBLE, x + w, y + (dy++) * dh, w, h, hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "mse: ", WS_CHILD | WS_VISIBLE, x, y + (dy) * dh, w, h, hwnd, NULL, NULL, NULL);

	GUI.mse = CreateWindowA("static", "", WS_CHILD | WS_VISIBLE, x + w, y + (dy++) * dh, w, h, hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "win rate: ", WS_CHILD | WS_VISIBLE, x, y + (dy) * dh, w, h, hwnd, NULL, NULL, NULL);

	GUI.winHate = CreateWindowA("static", "", WS_CHILD | WS_VISIBLE, x + w, y + (dy++) * dh, w, h, hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "imps: ", WS_CHILD | WS_VISIBLE, x, y + (dy) * dh, w, h, hwnd, NULL, NULL, NULL);

	GUI.imps = CreateWindowA("static", "", WS_CHILD | WS_VISIBLE, x + w, y + (dy++) * dh, w, h, hwnd, NULL, NULL, NULL);
	GUI.rede = CreateWindowA("static", "", WS_CHILD | WS_VISIBLE, x, y + (dy++) * dh, 1000, 1000, hwnd, NULL, NULL, NULL);

*/
}

int getFileName(char *fileName, int len) {
	OPENFILENAME ofn;       // common dialog box structure
	HWND hwnd = GUI.hmain;              // owner window
	HANDLE hf = NULL;              // file handle
// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = fileName;
// Set lpstrFile[0] to '\0' so that GetOpenFileName does not
// use the contents of szFile to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = len;
	ofn.lpstrFilter = "lua\0*.LUA\0All Files\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

// Display the Open dialog box.

	return GetOpenFileName(&ofn);
}

int dialogBox(char *title, char *msg) {
	int msgboxID = MessageBox(GUI.hmain, msg, title, MB_ICONEXCLAMATION | MB_YESNO);
	if (msgboxID == IDYES) {
		return 1;
	}
	return 0;
}

void getLuaFILE(char *dst, int len, int nargs, const char **args) {
	if (nargs != 2) {
		while (!getFileName(dst, len))
			if (!dialogBox("Nenhum arquivo selecionado", "Deseja tentar novamente?")) { exit(GAB_FAILED_OPEN_FILE); }
		return;
	}
	snprintf(dst, len, "%s", args[1]);
}

int CaptureAnImage() {
	HWND hWnd = GUI.hmain;
	char *fileName = GUI.arg;
	HDC hdcScreen;
	HDC hdcWindow;
	HDC hdcMemDC = NULL;
	HBITMAP hbmScreen = NULL;
	BITMAP bmpScreen;
	DWORD dwBytesWritten = 0;
	DWORD dwSizeofDIB = 0;
	HANDLE hFile = NULL;
	char *lpbitmap = NULL;
	HANDLE hDIB = NULL;
	DWORD dwBmpSize = 0;

	// Retrieve the handle to a display device context for the client
	// area of the window.
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hWnd, &ps);
	hdcScreen = GetDC(NULL);
	hdcWindow = GetDC(hWnd);

	// Create a compatible DC, which is used in a BitBlt from the window DC.
	hdcMemDC = CreateCompatibleDC(hdcWindow);

	if (!hdcMemDC) {
		MessageBoxW(hWnd, L"CreateCompatibleDC has failed", L"Failed", MB_OK);
		goto done;
	}

	// Get the client area for size calculation.
	RECT rcClient;
	GetClientRect(hWnd, &rcClient);

	// This is the best stretch mode.
	SetStretchBltMode(hdcWindow, HALFTONE);

	// The source DC is the entire screen, and the destination DC is the current window (HWND).
	if (!StretchBlt(hdcWindow, 0, 0, rcClient.right, rcClient.bottom, hdcScreen, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), SRCCOPY)) {
		MessageBoxW(hWnd, L"StretchBlt has failed", L"Failed", MB_OK);
		goto done;
	}

	// Create a compatible bitmap from the Window DC.
	hbmScreen = CreateCompatibleBitmap(hdcWindow, rcClient.right - rcClient.left, rcClient.bottom - rcClient.top);

	if (!hbmScreen) {
		MessageBoxW(hWnd, L"CreateCompatibleBitmap Failed", L"Failed", MB_OK);
		goto done;
	}

	// Select the compatible bitmap into the compatible memory DC.
	SelectObject(hdcMemDC, hbmScreen);

	// Bit block transfer into our compatible memory DC.
	if (!BitBlt(hdcMemDC, 0, 0, rcClient.right - rcClient.left, rcClient.bottom - rcClient.top, hdcWindow, 0, 0, SRCCOPY)) {
		MessageBoxW(hWnd, L"BitBlt has failed", L"Failed", MB_OK);
		goto done;
	}

	// Get the BITMAP from the HBITMAP.
	GetObject(hbmScreen, sizeof(BITMAP), &bmpScreen);

	BITMAPFILEHEADER bmfHeader;
	BITMAPINFOHEADER bi;

	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = bmpScreen.bmWidth;
	bi.biHeight = bmpScreen.bmHeight;
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	dwBmpSize = ((bmpScreen.bmWidth * bi.biBitCount + 31) / 32) * 4 * bmpScreen.bmHeight;

	// Starting with 32-bit Windows, GlobalAlloc and LocalAlloc are implemented as wrapper functions that
	// call HeapAlloc using a handle to the process's default heap. Therefore, GlobalAlloc and LocalAlloc
	// have greater overhead than HeapAlloc.
	hDIB = GlobalAlloc(GHND, dwBmpSize);
	lpbitmap = (char *) GlobalLock(hDIB);

	// Gets the "bits" from the bitmap, and copies them into a buffer
	// that's pointed to by lpbitmap.
	GetDIBits(hdcWindow, hbmScreen, 0, (UINT) bmpScreen.bmHeight, lpbitmap, (BITMAPINFO *) &bi, DIB_RGB_COLORS);

	// A file is created, this is where we will save the screen capture.
	hFile = CreateFileA(fileName, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	// Add the size of the headers to the size of the bitmap to get the total file size.
	dwSizeofDIB = dwBmpSize + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	// Offset to where the actual bitmap bits start.
	bmfHeader.bfOffBits = (DWORD) sizeof(BITMAPFILEHEADER) + (DWORD) sizeof(BITMAPINFOHEADER);

	// Size of the file.
	bmfHeader.bfSize = dwSizeofDIB;

	// bfType must always be BM for Bitmaps.
	bmfHeader.bfType = 0x4D42; // BM.

	WriteFile(hFile, (LPSTR) &bmfHeader, sizeof(BITMAPFILEHEADER), &dwBytesWritten, NULL);
	WriteFile(hFile, (LPSTR) &bi, sizeof(BITMAPINFOHEADER), &dwBytesWritten, NULL);
	WriteFile(hFile, (LPSTR) lpbitmap, dwBmpSize, &dwBytesWritten, NULL);

	// Unlock and Free the DIB from the heap.
	GlobalUnlock(hDIB);
	GlobalFree(hDIB);

	// Close the handle for the file that was created.
	CloseHandle(hFile);

	// Clean up.
	done:
	DeleteObject(hbmScreen);
	DeleteObject(hdcMemDC);
	ReleaseDC(NULL, hdcScreen);
	ReleaseDC(hWnd, hdcWindow);
	EndPaint(hWnd, &ps);
	return 0;
}

#endif //GAB_UI_H
