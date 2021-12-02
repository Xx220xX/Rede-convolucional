//
// Created by Henrique on 01/12/2021.
//

#ifndef GAB_UI_H
#define GAB_UI_H

#include "windows.h"
#include "error_list.h"
#include "Thread.h"

#include <wchar.h>
#include <strsafe.h>

#define cnnMain _local_main_

int cnnMain(int,  char **);


LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

void CreateLabels(HWND);

typedef HWND TextLabel;

struct {
	TextLabel epoca;
	TextLabel imagem;
	TextLabel mse;
	TextLabel winHate;
	TextLabel imps;
	TextLabel status;
	atomic_int *can_run;
	atomic_int *force_end;
	HWND hmain;

	void (*setText)(TextLabel text, const char *format, ...);

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

int wstrlen(const LPWSTR lpwstr) {
	int len = 0;
	for (; lpwstr[len]; ++len);
	return len;
}

void run_main(PSTR pCmdLine) {

	cnnMain(__argc , __argv);


	PostMessageA(GUI.hmain, WM_DESTROY, 0, 0);

}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
				   PSTR pCmdLine, int CmdShow) {
	HWND hwnd;
	MSG msg;
	GUI.setText = GUI_setText;
	WNDCLASSW wc = {0};
	wc.lpszClassName = L"GabCnn";
	wc.hInstance = hInstance;
	wc.hbrBackground = GetSysColorBrush(COLOR_3DFACE);
	wc.lpfnWndProc = WndProc;
	wc.hCursor = LoadCursor(0, IDC_ARROW);

	RegisterClassW(&wc);
	hwnd = CreateWindowW(wc.lpszClassName, L"GabCnn",
						 WS_OVERLAPPEDWINDOW | WS_VISIBLE,
						 150, 150, 500, 500, 0, 0, hInstance, 0);
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
	}
	Thread_Release(hmain);
	return (int) msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {

#define BUF_LEN 10
	wchar_t buf[BUF_LEN];

	RECT rect;

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
	CreateWindowA("static", "epoca: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "status: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.status = CreateWindowA("static", "NAN",
							   WS_CHILD | WS_VISIBLE,
							   x + w, y + (dy++) * dh, 1000, h,
							   hwnd, NULL, NULL, NULL);
	GUI.epoca = CreateWindowA("static", "0",
							  WS_CHILD | WS_VISIBLE,
							  x + w, y + (dy++) * dh, w, h,
							  hwnd, NULL, NULL, NULL);

	CreateWindowA("static", "imagem: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.imagem = CreateWindowA("static", "0",
							   WS_CHILD | WS_VISIBLE,
							   x + w, y + (dy++) * dh, w, h,
							   hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "mse: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.mse = CreateWindowA("static", "NAN",
							WS_CHILD | WS_VISIBLE,
							x + w, y + (dy++) * dh, w, h,
							hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "win rate: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.winHate = CreateWindowA("static", "150",
								WS_CHILD | WS_VISIBLE,
								x + w, y + (dy++) * dh, w, h,
								hwnd, NULL, NULL, NULL);
	CreateWindowA("static", "imps: ",
				  WS_CHILD | WS_VISIBLE,
				  x, y + (dy) * dh, w, h,
				  hwnd, NULL, NULL, NULL);

	GUI.imps = CreateWindowA("static", "NAN",
							 WS_CHILD | WS_VISIBLE,
							 x + w, y + (dy++) * dh, w, h,
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
			if (!dialogBox("Nenhum arquivo selecionado", "Deseja tentar novamente?"))exit(FAILED_OPEN_FILE);
		return;
	}
	snprintf(dst, len, "%s", args[1]);
}

#endif //GAB_UI_H
