//
// Created by Henrique on 19/01/2022.
//

#ifndef MAIN_C_LOADFILE_H
#define MAIN_C_LOADFILE_H

int dialogBox(char *title, char *msg) {
	int msgboxID = MessageBox(GUI.hmain, msg, title, MB_ICONEXCLAMATION | MB_YESNO);
	if (msgboxID == IDYES) {
		return 1;
	}
	return 0;
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

void getLuaFILE(char *dst, int len, int nargs, const char **args) {
	if (nargs != 2) {
		while (!getFileName(dst, len))
			if (!dialogBox("Nenhum arquivo selecionado", "Deseja tentar novamente?")) {
				exit(GAB_FAILED_OPEN_FILE);
			}
		return;
	}
	snprintf(dst, len, "%s", args[1]);
}

#endif //MAIN_C_LOADFILE_H
