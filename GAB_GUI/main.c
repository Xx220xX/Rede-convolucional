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

#include <wchar.h>

#include <strsafe.h>
#include <commctrl.h>
#include <locale.h>
#include <stdint.h>

typedef struct {
} *Tensor;

typedef struct Cnn_t {
	///  versão da compilação
	const char *version;
	/// modo
	int mode;
	/// não pode mais adicionar camada
	int lock;

	/// constrolar quando ela está em processamento ou não
	int running;

	/// entrada da rede
	Tensor entrada;
	/// Ultimo gradiente
	Tensor ds;
	/// Tensor de objetivo da rede
	Tensor target;
	/// numero de camadas
	size_t l;
	/// camadas
	Camada *cm;
	/// kernels internos
	void *kernels;
	/// Lua vm
	void *LuaVm;

	void (*releaseL)(void *L);

	/// entrada da rede
	const P3d size_in;
	Ecx ecx;

	Gpu gpu;
	int8_t release_gpu;
	Queue queue;

	// methods
	int (*setInput)(struct Cnn_t *self, size_t x, size_t y, size_t z);

	int (*release)(struct Cnn_t **selfp);

	/// retorna a dimensão da saída da rede
	P3d (*getSizeOut)(struct Cnn_t *self);

	void (*setMode)(struct Cnn_t *self, int isTrainig);

	/// retorna a rede em json
	char *(*json)(struct Cnn_t *self, int showValue);

	void (*jsonF)(struct Cnn_t *self, int showValue, const char *fileName);

	int (*save)(struct Cnn_t *self, const char *filename);

	int (*load)(struct Cnn_t *self, const char *filename);

	int (*predict)(struct Cnn_t *self, Tensor input);

	int (*predictv)(struct Cnn_t *self, REAL *input);

	int (*learn)(struct Cnn_t *self, Tensor target);

	int (*learnv)(struct Cnn_t *self, REAL *target);

	int (*learnBatch)(struct Cnn_t *self, Tensor target, size_t batchSize);

	int (*fixBatch)(struct Cnn_t *self);

	int (*updateHitLearn)(struct Cnn_t *self, size_t iter);

	REAL (*mse)(struct Cnn_t *self);

	REAL (*mseT)(struct Cnn_t *self, Tensor target);

	int (*maxIndex)(struct Cnn_t *self);

	void (*print)(struct Cnn_t *self, const char *comment);

	void (*fprint)(struct Cnn_t *self, FILE *fdst, const char *comment);

	char *(*printstr)(struct Cnn_t *self, const char *comment);

	int (*normalizeIMAGE)(struct Cnn_t *self, Tensor dst_real, Tensor src_char);

	int (*extractVectorLabelClass)(struct Cnn_t *self, Tensor dst, Tensor label);

	int (*ConvolucaoF)(struct Cnn_t *self, P2d passo, P3d filtro, FAtivacao_t funcaoAtivacao, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right, Parametros p, RandomParams filtros);


	int (*Pooling)(struct Cnn_t *self, P2d passo, P2d filtro, uint32_t type);

	int (*Relu)(struct Cnn_t *self, REAL fator_menor0, REAL fator_maior0);

	int (*PRelu)(struct Cnn_t *self, Parametros params, RandomParams rdp_a);

	int (*FullConnect)(struct Cnn_t *self, size_t numero_neuronios, Parametros p, FAtivacao_t funcaoAtivacao, RandomParams rdp_pesos, RandomParams rdp_bias);

	int (*Padding)(struct Cnn_t *self, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);

	int (*DropOut)(struct Cnn_t *self, REAL probabilidadeSaida, cl_ulong seed);

	int (*SoftMax)(struct Cnn_t *self, int8_t flag);

	int (*BatchNorm)(struct Cnn_t *self, size_t batch_size, REAL epsilon, Parametros p, RandomParams randY, RandomParams randB);

	void (*removeLastLayer)(struct Cnn_t *self);

} *Cnn, Cnn_t;

extern const char *Cnn_version();

extern Cnn Cnn_new();


int cnnMain(int, char **);

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);


void checkArgs();


void run_main(PSTR pCmdLine) {
	cnnMain(__argc, __argv);
//	PostMessageA(GUI.hmain, WM_DESTROY, 0, 0);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int CmdShow) {
	checkArgs();
	HWND hwnd;
	MSG msg;

	WNDCLASSW wc = {.lpszClassName = L"GabCnn", .hInstance = hInstance, .hbrBackground = GetSysColorBrush(COLOR_3DFACE), .lpfnWndProc = WndProc, .hCursor = LoadCursor(0, IDC_ARROW),};
	RegisterClassW(&wc);


	RECT windowSize = {0};
	GetWindowRect(GetDesktopWindow(), &windowSize);
	MoveWindow(GetConsoleWindow(), 0, 0, windowSize.right * 0.5, windowSize.bottom * 0.94, 1);

	hwnd = CreateWindowW(wc.lpszClassName, L"GabCnn", WS_OVERLAPPEDWINDOW | WS_VISIBLE, windowSize.right / 2, 1, windowSize.right / 2, windowSize.bottom, NULL, 0, hInstance, 0);
	char title[250];
	snprintf(title, 250, "Gab Cnn v %s", Cnn_version());
	SetWindowTextA(hwnd, title);

	if (!hwnd) {
		fprintf(stderr, "Falha ao criar janela\n");
		exit(-1);
	}
	GUI.avaliar = 1;
	GUI_init(hwnd);
	GUI.hisntance = hInstance;
	GUI.menu = CreateMenu();
	GUI.menu_fitnes_option = CreateMenu();
	if (GUI.avaliar) {
		AppendMenuA(GUI.menu_fitnes_option, MF_BYCOMMAND | MF_CHECKED, IDM_FITNES, "Avaliar durante o treinamento");
	} else {
		AppendMenuA(GUI.menu_fitnes_option, MF_BYCOMMAND | MF_UNCHECKED, IDM_FITNES, "Avaliar durante o treinamento");
	}
	AppendMenuW(GUI.menu, MF_POPUP, (UINT_PTR) GUI.menu_fitnes_option, L"&View");
	SetMenu(GUI.hmain, GUI.menu);

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
