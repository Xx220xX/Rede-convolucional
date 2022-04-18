#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN

#include <windows.h>

HWND g_hWnd;
HINSTANCE g_hInstance;
static const int g_nWindowWidth = 600;
static const int g_nWindowHeight = 400;

// Used to double buffer the window
HDC hdcDisplay;
HDC hdcMemory;
HBITMAP hbmBackBuffer;
HBITMAP hbmOld;
RECT rClient;

// Image to display on screen
HDC hdcBitmap;
HBITMAP hbmBitmap;
HBITMAP hbmBitmapOld;
BITMAP bmBitmapInfo;

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

void CreateBackBuffer();

void DestroyBackBuffer();

void loadBitmap();

void UnloadBitmap();

void ClearScene();

void RenderScene();

void PresentScene();

void putIm(int pInt[1024], int i, int i1, HWND pHwnd, int i2, int i3, int i4, int i5);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow) {
	g_hInstance = hInstance;

	WNDCLASS wndclass;
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = WndProc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
	wndclass.lpszMenuName = 0;
	wndclass.lpszClassName = "AwesomeGDI";

	RegisterClass(&wndclass);

	SetRect(&rClient, 0, 0, g_nWindowWidth, g_nWindowHeight);
	AdjustWindowRect(&rClient, WS_OVERLAPPEDWINDOW | WS_VISIBLE, FALSE);

	g_hWnd = CreateWindow("AwesomeGDI", "Awesomium GDI Test", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
						  (GetSystemMetrics(SM_CXSCREEN) / 2) - ((rClient.right - rClient.left) / 2),
						  (GetSystemMetrics(SM_CYSCREEN) / 2) - ((rClient.right - rClient.left) / 2),
						  (rClient.right - rClient.left), (rClient.bottom - rClient.top), NULL, NULL, hInstance, szCmdLine);


//	CreateBackBuffer();
//	loadBitmap();

	MSG msg;
	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}


	return (int) msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	switch (iMsg) {
		case WM_CREATE:
			break;
		case WM_CLOSE:
			UnloadBitmap();
			DestroyBackBuffer();
			DestroyWindow(hwnd);
			break;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
		case WM_ERASEBKGND:
			return TRUE;
		case WM_PAINT:
			//BeginPaint(hwnd, &ps);
		{
			int im[32 * 32];
			for (int i = 0; i < 32 * 32; ++i) {
				im[i] = RGB(rand() % 256, rand() % 256, rand() % 256);
			}
			putIm(im, 32, 23, hwnd, 0, 0, 100, 100);
		}
//			ClearScene();
//			RenderScene();
//			PresentScene();
//
//			ValidateRect(hwnd, NULL);
			//EndPaint(hwnd, &ps);
			break;
	}
	return DefWindowProc(hwnd, iMsg, wParam, lParam);
}

void putIm(int *im, int w, int h, HWND pHwnd, int xDest, int yDest, int wDest, int hDest) {
	HDC hdc = GetDC(pHwnd);
	HDC hdc_mem = CreateCompatibleDC(hdc);
	HBITMAP hback = CreateCompatibleBitmap(hdc, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
	HBITMAP old = (HBITMAP) SelectObject(hdc_mem, hback);
	HBITMAP hbit = CreateDiscardableBitmap(hdc_mem, w, h);
	HDC hdc_bit = CreateCompatibleDC(hdc);
	BITMAP bm;

	SetBitmapBits(hbit, w * h * 4, im);

	SelectObject(hdc_bit, hbit);
	GetObject(hbit, sizeof(BITMAP), &bm);

	BitBlt(hdc_mem, 0, 0, bm.bmWidth, bm.bmHeight, hdc_bit, 0, 0, SRCCOPY);
	StretchBlt(hdc, xDest, yDest, wDest, hDest, hdc_mem, 0, 0, bm.bmWidth, bm.bmHeight, SRCCOPY);

	SelectObject(hdc_bit, old);
	ReleaseDC(pHwnd, hdc);
	ReleaseDC(pHwnd, hdc_mem);
	ReleaseDC(pHwnd, hdc_bit);
	DeleteObject(hbit);
	DeleteObject(hback);
}

void CreateBackBuffer() {
	hdcDisplay = GetDC(g_hWnd);
	hdcMemory = CreateCompatibleDC(hdcDisplay);
	hbmBackBuffer = CreateCompatibleBitmap(hdcDisplay, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
	hbmOld = (HBITMAP) SelectObject(hdcMemory, hbmBackBuffer);
}

void DestroyBackBuffer() {
	SelectObject(hdcMemory, hbmOld);
	DeleteObject(hbmBackBuffer);
	DeleteDC(hdcMemory);
	ReleaseDC(g_hWnd, hdcDisplay);
}

void loadBitmap() {
	hbmBitmap = (HBITMAP) LoadImage(g_hInstance, "a.bmp", IMAGE_BITMAP, 0, 0, LR_DEFAULTCOLOR | LR_LOADFROMFILE);
	hdcBitmap = CreateCompatibleDC(hdcDisplay);
	hbmBitmapOld = (HBITMAP) SelectObject(hdcBitmap, hbmBitmap);
	GetObject(hbmBitmap, sizeof(BITMAP), &bmBitmapInfo);
}

void UnloadBitmap() {
	SelectObject(hdcBitmap, hbmBitmapOld);
	DeleteObject(hbmBitmap);
	DeleteDC(hdcBitmap);
}

void ClearScene() {
	GetClientRect(g_hWnd, &rClient);
}

void RenderScene() {
	BitBlt(hdcMemory, 0, 0, bmBitmapInfo.bmWidth, bmBitmapInfo.bmHeight, hdcBitmap, 0, 0, SRCCOPY);
}

void PresentScene() {
	StretchBlt(hdcDisplay, 0, 0, rClient.right, rClient.bottom, hdcMemory, 0, 0, bmBitmapInfo.bmWidth, bmBitmapInfo.bmHeight, SRCCOPY);
}