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

#define cnnMain _local_main_


int cnnMain(int, char **);

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

void CreateLabels(HWND);

void checkArgs();


void run_main(PSTR pCmdLine) {
//	Client_connect();
	cnnMain(__argc, __argv);
//	Client_close();
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
//	/Figure_new(&GUI.f, hwnd, hInstance, 30, 30, 400, 400);
//	GUI.f.bkcolor = RGB(0xff, 0xff, 0xff);
//	GUI.f.xstep = 0.5;
//	GUI.f.ystep = 0.5;
//	GUI.f.grid = 1;
//	MoveWindow(GUI.f.window,50,50,300,300,1);
//	GUI.f.putAxe(&GUI.f, RGB(0xff,0,0));

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
			printf("versão %s\n", "0");
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
	}

	return DefWindowProcW(hwnd, msg, wParam, lParam);
}


void CreateLabels(HWND hwnd) {

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
			if (!dialogBox("Nenhum arquivo selecionado", "Deseja tentar novamente?")) {
				exit(GAB_FAILED_OPEN_FILE);
			}
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
	RECT myrec;
	GetWindowRect(hWnd, &myrec);
	if (!StretchBlt(hdcWindow, 0, 0, rcClient.right, rcClient.bottom, hdcScreen, myrec.left, myrec.top, myrec.right, myrec.bottom, SRCCOPY)) {
		MessageBoxW(hWnd, L"StretchBlt has failed", L"Failed", MB_OK);
		goto done;
	}
//	if (!StretchBlt(hdcWindow, 0, 0, rcClient.right, rcClient.bottom, hdcScreen, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), SRCCOPY)) {
//		MessageBoxW(hWnd, L"StretchBlt has failed", L"Failed", MB_OK);
//		goto done;
//	}

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


#include "math.h"

int cnnMaint(int arg, char **args) {
	GUI.make_train();
}

#endif //GAB_UI_H
