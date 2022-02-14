//
// Created by Henrique on 01/12/2021.
//

#ifndef GAB_UI_H
#define GAB_UI_H
#pragma comment(lib, "user32")
#pragma comment(lib, "gdi32.lib")

#include "windows.h"
#include "windowsx.h"
#include "thread/Thread.h"
#include <error_list.h>
#include <wchar.h>

#include <strsafe.h>
#include <commctrl.h>
#include <locale.h>
#include <stdint.h>
#include "GUI.h"
#include "plot.h"
#include "utils/loadFile.h"
#include "utils/captureScreen.h"

#define cnnMain _local_main_


int cnnMain(int, char **);

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);


void checkArgs();


void run_main(PSTR pCmdLine) {
	cnnMain(__argc, __argv);
	PostMessageA(GUI.hmain, WM_DESTROY, 0, 0);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int CmdShow) {
	checkArgs();
	HWND hwnd;
	MSG msg;

	WNDCLASSW wc = {.lpszClassName = L"GabCnn", .hInstance = hInstance, .hbrBackground = GetSysColorBrush(COLOR_3DFACE), .lpfnWndProc = WndProc, .hCursor = LoadCursor(0, IDC_ARROW),};
	RegisterClassW(&wc);
	Figure_INITW(hInstance);

	RECT windowSize = {0};
	GetWindowRect(GetDesktopWindow(), &windowSize);
	MoveWindow(GetConsoleWindow(), 0, 0, windowSize.right * 0.5, windowSize.bottom * 0.94, 1);

	hwnd = CreateWindowW(wc.lpszClassName, L"GabCnn", WS_OVERLAPPEDWINDOW | WS_VISIBLE, windowSize.right / 2, 1, windowSize.right / 2, windowSize.bottom, NULL, 0, hInstance, 0);
	SetWindowTextA(hwnd, "Gab Cnn");

	if (!hwnd) {
		fprintf(stderr, "Falha ao criar janela\n");
		exit(-1);
	}
	GUI_init(hwnd);
	GUI.hisntance = hInstance;
	GUI.menu = CreateMenu();
	GUI.menu_fitnes_option = CreateMenu();
	AppendMenuA(GUI.menu_fitnes_option, MF_BYCOMMAND | MF_CHECKED, IDM_FITNES, "Avaliar durante o treinamento");
	AppendMenuW(GUI.menu, MF_POPUP, (UINT_PTR) GUI.menu_fitnes_option, L"&View");
	SetMenu(GUI.hmain, GUI.menu);
	GUI.avaliar = 1;

	HICON hIcon = 0;
	hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(11));
	if (!hIcon) {
		hIcon = LoadImageW(hInstance, L"icone.ico", IMAGE_ICON, 256, 256,    // or whatever size icon you want to load
						   LR_DEFAULTCOLOR | LR_LOADFROMFILE);
	}
	if (hIcon) {
		SendMessageA(hwnd, WM_SETICON, ICON_BIG, (LPARAM) hIcon);
		SendMessageA(hwnd, WM_SETICON, ICON_SMALL, (LPARAM) hIcon);
		SendMessageA(hwnd, WM_SETICON, ICON_SMALL2, (LPARAM) hIcon);

	}
	Handle hmain = Thread_new(run_main, pCmdLine);
	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);

	}
	Thread_Release(hmain);
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
		}else if(!strcmp(args[i], "-update") || !strcmp(args[i], "--update")){
			printf("Gabriela IA\n");
			printf("versão %s\n", "0");
			printf("https://xx220xx.github.io/Rede-convolucional/\n\n");
			printf("Buscando atualizações\n");
			FILE *f = fopen("gab_version.py","w");
			fprintf(f,"version = '%s'\n", Cnn_version());
			fclose(f);
			system("start \"python update.py\"");
			exit(0);
		}
	}
	if (nargs > 2) {
		fprintf(stderr, "Argumentos invalidos\n");
		exit(GAB_INVALID_PARAM);
	}
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	switch (msg) {
		case WM_CREATE:
			GUI.hmain = hwnd;
			GUI.status = CreateWindowA("static", "Gabriela iniciada", WS_CHILD | WS_VISIBLE, 1, 1, 200, 20, GUI.hmain, NULL, NULL, NULL);
			GUI_addLabel("Pronta para aprender", 1, 20, 200, 20);
			EnableMenuItem(GetSystemMenu(GetConsoleWindow(), FALSE), SC_CLOSE, MF_BYCOMMAND | MF_DISABLED | MF_GRAYED);
			break;
		case WM_UPDATEUISTATE:
			if (lParam == GUI_UPDATE_WINDOW) {

				((void (*)()) wParam)();
			}
			break;

		case WM_CLOSE:
			if (MessageBoxA(hwnd, "Really quit?", "Gab", MB_OKCANCEL) == IDOK) {
				if (GUI.can_run && GUI.force_end) {
					*GUI.can_run = 0;
					*GUI.force_end = 1;
					return 0;
				}
				break;
			}
			return 0;// User canceled. Do nothing.
		case WM_PAINT:
			GUI.endDraw = 1;
			break;
		case WM_MOVE:
		case WM_SIZE: {
			RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
			break;
		}
		case WM_DESTROY:
			GUI_releasefigs();
			PostQuitMessage(0);
			break;
		case WM_COMMAND:
			if (wParam == IDM_FITNES) {
				GUI.avaliar = !GUI.avaliar;
				if (GUI.avaliar) {
					CheckMenuItem(GUI.menu, IDM_FITNES, MF_BYCOMMAND | MF_CHECKED);
				} else {
					CheckMenuItem(GUI.menu, IDM_FITNES, MF_BYCOMMAND | MF_UNCHECKED);
				}
			}
			break;
	}
	return DefWindowProcW(hwnd, msg, wParam, lParam);
}


int cnnMain2(int arg, char **args) {
	GUI.make_train();
	Sleep(100);
	while (1);
	return 0;
}

#endif //GAB_UI_H
