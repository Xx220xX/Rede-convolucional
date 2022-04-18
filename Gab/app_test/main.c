//
// Created by Henrique on 13/04/2022.
//

#pragma comment(lib, "user32")
#pragma comment(lib, "gdi32.lib")

#include "windows.h"
#include "windowsx.h"
#include "thread/Thread.h"
#include <error_list.h>
#include <wchar.h>
#include <stdio.h>
#include <commctrl.h>
#include "cnn/cnn_lua.h"
#include "ui_image.h"
#include "Image.h"

#define RECT_PRINT(rect) printf("%ld %ld %ld %ld\n", rect.left, rect.right, rect.top, rect.bottom);

#define BT_ABRIR 0x01
#define LIST_IMAGES 0x02
typedef struct {
	int drawing;
	POINT ptPrevious;
} DrawWindow;
struct {
	HWND main_win;
	HINSTANCE hinstance;
	HWND b_abrir;
	HWND window_open;
	HWND l_imagens;
	DrawWindow dwin;
	Cnn cnn;
	HMENU hmenu;
	char cnnfile[250];
	char imagefile[250];
	Image image;
} APP = {0};


LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK Draw_window_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

void loadImage();

int Draw_window(HINSTANCE hInsta) {
	WNDCLASSEX wc;
	wc.hInstance = hInsta; // inj_hModule;
	wc.lpszClassName = "Draw_window";
	wc.lpfnWndProc = Draw_window_proc;
	wc.style = CS_DBLCLKS;
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.lpszMenuName = NULL;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hbrBackground = (HBRUSH) CreateSolidBrush(RGB(255, 0, 0));
	return RegisterClassEx(&wc);
}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int CmdShow) {
	HWND main_window;
	MSG msg;
	APP.hinstance = hInstance;
	WNDCLASSW wc = {.lpszClassName = L"Gab Check", .hInstance = hInstance, .hbrBackground = GetSysColorBrush(COLOR_3DFACE), .lpfnWndProc = WndProc, .hCursor = LoadCursor(0, IDC_ARROW),};
	RegisterClassW(&wc);
	Draw_window(hInstance);
	RECT windowSize = {0};
	GetWindowRect(GetDesktopWindow(), &windowSize);

	main_window = CreateWindowW(wc.lpszClassName, L"Gab Check", WS_OVERLAPPEDWINDOW | WS_VISIBLE, windowSize.right / 6, 1, windowSize.right * (4.0 / 6), windowSize.bottom * 0.95, NULL, 0, hInstance, 0);
	APP.main_win = main_window;
	char title[250];
	snprintf(title, 250, "Gab Check v %s", Cnn_version());
	SetWindowTextA(main_window, title);

	if (!main_window) {
		fprintf(stderr, "Falha ao criar janela\n");
		exit(-1);
	}
	HICON hIcon = 0;
	hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(11));

	if (!hIcon) {
		hIcon = LoadImageW(hInstance, L"icone.ico", IMAGE_ICON, 256, 256,    // or whatever size icon you want to load
						   LR_DEFAULTCOLOR | LR_LOADFROMFILE);
	}
	if (hIcon) {
		SendMessageA(main_window, WM_SETICON, ICON_BIG, (LPARAM) hIcon);
		SendMessageA(main_window, WM_SETICON, ICON_SMALL, (LPARAM) hIcon);
		SendMessageA(main_window, WM_SETICON, ICON_SMALL2, (LPARAM) hIcon);
	}




//	Handle hmain = Thread_new(run_main, pCmdLine);
	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);

	}
//	Thread_Release(hmain);
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
		} else if (!strcmp(args[i], "-update") || !strcmp(args[i], "--update")) {
			printf("Gabriela IA\n");
			printf("versão %s\n", "0");
			printf("https://xx220xx.github.io/Rede-convolucional/\n\n");
			printf("Buscando atualizações\n");
			FILE *f = fopen("gab_version.py", "w");
			fprintf(f, "version = '%s'\n", Cnn_version());
			fclose(f);
			f = fopen("update.py", "w");
			fprintf(f, "import os\n"
					   "import sys\n"
					   "try:\n"
					   "\timport urllib.request\n"
					   "\t\n"
					   "\tif not ('-read' in sys.argv):\n"
					   "\t\tos.system('Gab.exe --updatepy')\n"
					   "\t\n"
					   "\tprint('hello from py')\n"
					   "\tdef download_progress_hook(count, blockSize, totalSize):\n"
					   "\t\tprint(\"\\rBaixando %%.2f%%%%\" %%((count) * blockSize / totalSize * 100,), end='')\n"
					   "\t\n"
					   "\t\n"
					   "\tnewslink = ('https://raw.githubusercontent.com/Xx220xX/Rede-convolucional/master/RELEASE/lastversion.py', 'lastversion.py')\n"
					   "\t\n"
					   "\tprint(f'Baixando {newslink[1]}')\n"
					   "\turllib.request.urlretrieve(*newslink, reporthook=download_progress_hook)\n"
					   "\tprint()\n"
					   "\t\n"
					   "\timport lastversion\n"
					   "\t\n"
					   "\t\n"
					   "\tos.remove(newslink[1])\n"
					   "\tos.remove('gab_version.py')\n"
					   "except Exception as e:\n"
					   "\tprint(e)\n"
					   "\tinput()");
			fclose(f);
			system("start python update.py -ready");
			exit(0);
		} else if (!strcmp(args[i], "-updatepy") || !strcmp(args[i], "--updatepy")) {
			printf("Gabriela IA\n");
			printf("versão %s\n", "0");
			printf("https://xx220xx.github.io/Rede-convolucional/\n\n");
			printf("Buscando atualizações\n");
			FILE *f = fopen("gab_version.py", "w");
			fprintf(f, "version = '%s'\n", Cnn_version());
			fclose(f);
			exit(0);
		}
	}
	if (nargs > 2) {
		fprintf(stderr, "Argumentos invalidos\n");
		exit(GAB_INVALID_PARAM);
	}
}

#define  IDM_FILE_LOADCNN 0x01
#define  IDM_FILE_OPEN_IMAGE 0x02
#define  IDM_FILE_QUIT 0x03

#define IDFILTER_LUA "lua\0*.LUA\0All Files\0*.*\0"
#define IDFILTER_CNN "cnn\0*.CNN\0All Files\0*.*\0"
#define IDFILTER_image "Image\0*.png;*.ppm\0All Files\0*.*\0"

void push_menu(HWND hwnd) {
	HMENU hMenubar;
	HMENU hMenu;

	hMenubar = CreateMenu();
	hMenu = CreateMenu();
	AppendMenuW(hMenu, MF_STRING, IDM_FILE_LOADCNN, L"&Load Cnn");
	AppendMenuW(hMenu, MF_STRING, IDM_FILE_OPEN_IMAGE, L"&Load Image");
	AppendMenuW(hMenu, MF_SEPARATOR, 0, NULL);
	AppendMenuW(hMenu, MF_STRING, IDM_FILE_QUIT, L"&Quit");

	AppendMenuW(hMenubar, MF_POPUP, (UINT_PTR) hMenu, L"&File");
	SetMenu(hwnd, hMenubar);
}

int getFileName(HWND hwnd, char *fileName, int len, char *filter) {
	OPENFILENAME ofn;       // common dialog box structure
	// owner window
	HANDLE hf = NULL;              // file handle
// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = fileName;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = len;
	ofn.lpstrFilter = filter; //;
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = "%USERPROFILE%";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
// Display the Open dialog box.
	return GetOpenFileName(&ofn);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	RECT windowSize;
	GetClientRect(hwnd, &windowSize);
	switch (msg) {
		case WM_CREATE:
			APP.main_win = hwnd;
			APP.window_open = CreateWindowW(L"Draw_window", NULL, WS_CHILD | WS_CLIPSIBLINGS | WS_VISIBLE, 5, 5, windowSize.right * (1.0 / 3) - 10, windowSize.bottom - 10, hwnd, NULL, NULL, NULL);
			push_menu(hwnd);
			break;
		case WM_COMMAND:
			switch (LOWORD(wParam)) {
				case IDM_FILE_LOADCNN:
					break;
				case IDM_FILE_OPEN_IMAGE:
					if (getFileName(hwnd, APP.imagefile, 250, IDFILTER_image)) {
						loadImage();
					}
					break;
				case IDM_FILE_QUIT:
					SendMessage(hwnd, WM_CLOSE, 0, 0);
					break;
			}
		case WM_UPDATEUISTATE:
//			if (lParam == GUI_UPDATE_WINDOW) {
//				((void (*)()) wParam)();
//			}
			break;

		case WM_CLOSE:
			MessageBeep(MB_ICONINFORMATION);
			if (MessageBoxA(hwnd, "Really quit?", "Gab", MB_OKCANCEL) == IDOK) {

				break;
			}
			return 0;// User canceled. Do nothing.
		case WM_PAINT:
			RedrawWindow(APP.window_open, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
			break;
//		case WM_MOVE:
		case WM_SIZE: {
//			RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
			MoveWindow(APP.window_open, 5, 5, windowSize.right * (1.0 / 3) - 10, windowSize.bottom - 10, 1);
			break;
		}
		case WM_DESTROY:
//			GUI_releasefigs();
			PostQuitMessage(0);
			break;
	}
	return DefWindowProcW(hwnd, msg, wParam, lParam);
}

void loadImage() {
	int len = strlen(APP.imagefile);
	int error = 0;
	if (APP.imagefile[len - 1] == 'm') {// ppm
		error = loadPPM(APP.imagefile, &APP.image);
	} else if (APP.imagefile[len - 1] == 'g') { // png
		error = -1;
	}
	if (!error) {
		if (APP.image.channels == 3) {
			putImRGB(APP.image.raw, APP.image.raw + APP.image.w * APP.image.h, APP.image.raw + 2 * APP.image.w * APP.image.h,
			APP.image.w,APP.image.h,APP.window_open,0,0,100,100);
		}else{
			putImGray(APP.image.raw,
					  APP.image.w,APP.image.h,APP.window_open,0,0,100,100);
		}
	}else{
		printf("%d\n",error);
	}

}


LRESULT Draw_window_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	RECT rect;

	GetClientRect(hwnd, &rect);

	switch (msg) {
		case WM_CREATE:
//			CreateWindowA("static","", WS_CHILD | WS_VISIBLE, x, y, w, h, GUI.hmain, NULL, NULL, NULL);
//			APP.b_abrir = CreateWindowW(L"BUTTON", L"Abrir", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON, 1, rect.bottom -50, rect.right/2, 50, hwnd, (HMENU) BT_ABRIR, APP.hinstance, NULL);
			break;
//		case WM_COMMAND:
//			if (((HWND) lParam) && (HIWORD(wParam) == BN_CLICKED)) {
//				switch (LOWORD(wParam)) {
//					case BT_ABRIR:
//						printf("Botão abrir\n");
//						#define W 40
//						#define H 40
//
//						int im[W * H] = {0};
//						for (int i = 0; i < H; ++i) {
//							for (int j = 0; j < W; ++j) {
//								im[i * W + j] = RGB(255.0*(i+j)/(W+H)*rand()/RAND_MAX,0,0);// RGB(rand() % 256, rand() % 256, rand() % 256);
//							}
//						}
//						putIm(im, W, H, hwnd, 1, rect.bottom / 2 + 50, rect.right - 2, rect.bottom / 2 - 50);
//						break;
//				}
//			}
			break;
		case WM_PAINT:
//			MoveWindow(APP.b_abrir, 1, rect.bottom / 2, rect.right - 2, 50, 1);

			break;
		case WM_LBUTTONDOWN:
			APP.dwin.drawing = TRUE;
			APP.dwin.ptPrevious.x = LOWORD(lParam);
			APP.dwin.ptPrevious.y = HIWORD(lParam);
			return 0L;

		case WM_LBUTTONUP:
			if (APP.dwin.drawing) {
				HDC hdc = GetDC(hwnd);
				MoveToEx(hdc, APP.dwin.ptPrevious.x, APP.dwin.ptPrevious.y, NULL);
				LineTo(hdc, LOWORD(lParam), HIWORD(lParam));
				ReleaseDC(hwnd, hdc);
			}
			APP.dwin.drawing = FALSE;
			return 0L;

		case WM_MOUSEMOVE:
			if (APP.dwin.drawing) {
				HDC hdc = GetDC(hwnd);
				MoveToEx(hdc, APP.dwin.ptPrevious.x, APP.dwin.ptPrevious.y, NULL);
				LineTo(hdc, APP.dwin.ptPrevious.x = LOWORD(lParam),
					   APP.dwin.ptPrevious.y = HIWORD(lParam));
				ReleaseDC(hwnd, hdc);
			}
			return 0L;

	}
	return DefWindowProcW(hwnd, msg, wParam, lParam);

}
