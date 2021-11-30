
#include <treino/dir.h>
#include <png/png.h>
#include "treino/Manage.h"
#include "windows.h"
#include "mainui.h"

#define LENGTH 500

int getFileName(char *fileName, int len);

int dialogBox(char *title, char *msg);


int menu(Manage *manage);

int main(int nargs, char **args) {
	int error;
	enableUtf8();
	char file[LENGTH] = {0};
	if (nargs != 2) {
		while (!getFileName(file, LENGTH))
			if (!dialogBox("Abrir arquivo", "Nenhum arquivo foi encontrado. Sair do programa?"))
				continue;
			else return 0;
	}
	Manage manage = Manage_new(file, 0);
	error = manage.cnn->erro->error;

//	treino:
//	Manage_train(&manage, 1);
//	Manage_loop(&manage, 0);
//
//	fitnes:
//	Manage_fitnes(&manage, 1);
//	Manage_loop(&manage, 0);
//	if (!manage.force_close) {
//		switch (menu(&manage)) {
//			case 1:
//				goto treino;
//				break;
//			case 2:
//				goto fitnes;
//				break;
//			case 3:
//				goto load;
//				break;
//			default:
//				break;
//
//		}
//	}
//
//
	Manage_release(&manage);
	return error;
}

int menu(Manage *manage) {
	int escolha = 8;
	double t0 = getus();
	printf("pressione qualquer tecla para abrir o menu:...\n");
	printf("Encerra em  s");
	int y = wherey();
	size_t len;
	char *tmp = NULL;
	while (getus() - t0 < 5000) {
		if (kbhit()) {
			escolha = 0;
			break;
		}
		gotoxy(12, y);
		printf("%d", (int) (5 - (getus() - t0) / 1e6));
	}
	while (escolha != 8) {
		manage->cnn->erro->error = 0;
		system("cls");
		gotoxy(1, 1);
		printf("[1] continuar treinamento\n");
		printf("[2] continuar avaliação \n");
		printf("[3] salvar rede\n");
		printf("[4] mostrar rede:\n");
		printf("[5] lua console:\n");
		printf("[6] resetar imagens\n");
		printf("[7] show graphic\n");
		printf("[8] encerrar\nopcao:");
		scanf("%u", &escolha);
		fflush(stdin);

		switch (escolha) {
			case 1:
				Manage_run(manage, 1);
				return 1;
			case 2:
				Manage_run(manage, 1);
				return 2;
			case 3:
				len = snprintf(NULL, 0, "redes/%s.cnn", manage->name) + 1;
				tmp = alloc_mem(len + 1, 1);
				snprintf(tmp, len, "redes/%s.cnn", manage->name);
				manage->cnn->save(manage->cnn, tmp);
				printf("salvado\n");
				free_mem(tmp);
				break;
			case 4:
				manage->cnn->print(manage->cnn, "//");
				system("pause");
				break;
			case 5:
				CnnLuaConsole(manage->cnn);
				break;
			case 6:
				manage->image = 0;
				break;
			case 7:
				system("python showGraphic.py");
				break;
			case 8:
				break;
			default:
				printf("invalid option");

		}

		Sleep(300);
	}

}

int getFileName(char *fileName, int len) {
	OPENFILENAME ofn;       // common dialog box structure
	HWND hwnd = GetActiveWindow();              // owner window
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
			NULL,
			msg,
			title,
			MB_ICONEXCLAMATION | MB_YESNO
	);
	if (msgboxID == IDYES) {
		return 1;
	}
	return 0;
}
