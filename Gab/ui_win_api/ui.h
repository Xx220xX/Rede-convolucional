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
//#undef REAL
//#include <gdiplus.h>

//#include "image_window.h"

#define cnnMain _local_main_
#define  USE_PROGRESS_BAR  0

int cnnMain(int, char **);


LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

void CreateLabels(HWND);

void checkArgs();

typedef HWND TextLabel;
typedef HWND ProgressBar[2];

struct {
	TextLabel epoca;
	TextLabel imagem;
	TextLabel mse;
	TextLabel winHate;
	TextLabel imps;
	TextLabel status;
	TextLabel rede;
	ProgressBar progress;
	atomic_int *can_run;
	atomic_int *force_end;
	HWND hmain;
	HWND hconsole;

	void (*setText)(TextLabel text, const char *format, ...);

	void (*setProgress)(ProgressBar progressBar, double value);
} GUI;

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

void GUI_setProgress(ProgressBar progressBar, double value) {
	int ivalue = value;
#if (USE_PROGRESS_BAR == 1)
	SendMessage(progressBar[0], PBM_SETPOS, (WPARAM) ivalue % 101, 0);
	GUI_setText(progressBar[1], "%.2lf%%", value);
#endif
}

int wstrlen(const LPWSTR lpwstr) {
	int len = 0;
	for (; lpwstr[len]; ++len);
	return len;
}

void run_main(PSTR pCmdLine) {
//	Client_connect();
	cnnMain(__argc, __argv);
//	Client_close();
	PostMessageA(GUI.hmain, WM_DESTROY, 0, 0);

}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
				   PSTR pCmdLine, int CmdShow) {



	checkArgs();
//	GdiplusStartupInput gdii;
//	ULONG_PTR gditoken;
//	GdiplusStartup(&gditoken,&gdii,NULL);

	GUI.hconsole = FindWindowA("ConsoleWindowClass", NULL);
	HWND hwnd;
	MSG msg;
	GUI.setText = GUI_setText;
	GUI.setProgress = GUI_setProgress;
	WNDCLASSW wc = {0};
	wc.lpszClassName = L"GabCnn";
	wc.hInstance = hInstance;
	wc.hbrBackground = GetSysColorBrush(COLOR_3DFACE);
	wc.lpfnWndProc = WndProc;
	wc.hCursor = LoadCursor(0, IDC_ARROW);
	RegisterClassW(&wc);
	hwnd = CreateWindowW(wc.lpszClassName, L"GabCnn",
						 WS_OVERLAPPEDWINDOW | WS_VISIBLE,
						 150, 150, 750, 500, GUI.hconsole, 0, hInstance, 0);
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
			CreateLabels(hwnd);
			break;
		case WM_MOVE:

//			GetWindowRect(hwnd, &rect);
//
//			StringCbPrintfW(buf, BUF_LEN, L"%ld", rect.left);
//			SetWindowTextW(hwndSta1, buf);
//
//			StringCbPrintfW(buf, BUF_LEN, L"%ld", rect.top);
//			SetWindowTextW(hwndSta2, buf);
			break;
		case WM_CLOSE:
			if (MessageBoxA(hwnd, "Really quit?", "Gab", MB_OKCANCEL) == IDOK) {
				*GUI.can_run = 0;
				*GUI.force_end = 1;
			}
			// Else: User canceled. Do nothing.
			return 0;
		case WM_PAINT:
//			printf("\tWM_PAINT\n");
//			OnShowImg(hwnd,wParam,lParam)

			break;
//		case WM_CTLCOLORSTATIC:
			// Set the colour of the text for our URL
//			if (((HWND)lParam == GetDlgItem(GUI.progress[1], 0))){
			// set the text colour in (HDC)lParam
//				SetBkMode((HDC) GetDC(GUI.progress[1]),TRANSPARENT);
//				SetTextColor((HDC) GetDC(GUI.progress[1]), RGB(255,0,0));
//				 NOTE: per documentation as pointed out by selbie, GetSolidBrush would leak a GDI handle.
//				return (BOOL)GetSysColorBrush(COLOR_MENU);
//			}


		case WM_DESTROY:
			PostQuitMessage(0);
			break;
	}

	return DefWindowProcW(hwnd, msg, wParam, lParam);
}

void CreateLabels(HWND hwnd) {
	int x = 1, h = 25, w = 150, dy = 0;
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
	CreateWindowA("static", "status: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.status = CreateWindowA("static", "",
							   WS_CHILD | WS_VISIBLE,
							   x + w, y + (dy++) * dh, 1000, h,
							   hwnd, NULL, NULL, NULL);

	CreateWindowA("static", "epoca: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);
	GUI.epoca = CreateWindowA("static", "",
							  WS_CHILD | WS_VISIBLE,
							  x + w, y + (dy++) * dh, w, h,
							  hwnd, NULL, NULL, NULL);

	CreateWindowA("static", "imagem: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.imagem = CreateWindowA("static", "",
							   WS_CHILD | WS_VISIBLE,
							   x + w, y + (dy++) * dh, w, h,
							   hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "mse: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.mse = CreateWindowA("static", "",
							WS_CHILD | WS_VISIBLE,
							x + w, y + (dy++) * dh, w, h,
							hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "win rate: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.winHate = CreateWindowA("static", "",
								WS_CHILD | WS_VISIBLE,
								x + w, y + (dy++) * dh, w, h,
								hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "imps: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.imps = CreateWindowA("static", "",
							 WS_CHILD | WS_VISIBLE,
							 x + w, y + (dy++) * dh, w, h,
							 hwnd, NULL, NULL, NULL);
	GUI.rede = CreateWindowA("static", "",
							 WS_CHILD | WS_VISIBLE,
							 x, y + (dy++) * dh, 1000, 1000,
							 hwnd, NULL, NULL, NULL);


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
	int msgboxID = MessageBox(
			GUI.hmain,
			msg,
			title,
			MB_ICONEXCLAMATION | MB_YESNO
	);
	if (msgboxID == IDYES) {
		return 1;
	}
	return 0;
}

void getLuaFILE(char *dst, int len, int nargs, const char **args) {
	if (nargs != 2) {
		while (!getFileName(dst, len))
			if (!dialogBox("Nenhum arquivo selecionado", "Deseja tentar novamente?"))exit(GAB_FAILED_OPEN_FILE);
		return;
	}
	snprintf(dst, len, "%s", args[1]);
}

#endif //GAB_UI_H
