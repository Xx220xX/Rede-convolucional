//
// Created by Henrique on 13/04/2022.
//

#ifndef MAIN_C_UI_IMAGE_H
#define MAIN_C_UI_IMAGE_H

#include <windows.h>

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
void putImGray(unsigned char *im, int w, int h, HWND pHwnd, int xDest, int yDest, int wDest, int hDest) {
	int nim[w*h];
	for (int i = w * h-1; i >=0 ; --i) {
		nim[i] = RGB(im[i],im[i],im[i]);
	}
	putIm(nim,w,h,pHwnd,xDest,yDest,wDest,hDest);
}
void putImRGB(unsigned char *r, unsigned char *g,unsigned char *b, int w, int h, HWND pHwnd, int xDest, int yDest, int wDest, int hDest) {
	int nim[w*h];
	for (int i = w * h-1; i >=0 ; --i) {
		nim[i] = RGB(r[i],g[i],b[i]);
	}
	putIm(nim,w,h,pHwnd,xDest,yDest,wDest,hDest);
}

#endif //MAIN_C_UI_IMAGE_H
