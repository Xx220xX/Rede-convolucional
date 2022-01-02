//
// Created by Henrique on 29/12/2021.
//

#ifndef GAB_GRAFICO_H
#define GAB_GRAFICO_H
#include "windows.h"
typedef struct {
	int x, y;
	int w, h;
	double xmin,ymin,xmax,ymax;
	POINT *points;
	int npoint;
	DWORD color;
	char title[30];
} Graph_area;

void addPoint(Graph_area *self, double x, double y) {
	self->npoint++;
	self->points = realloc(self->points, self->npoint * sizeof(POINT));
	x = self->x + x / (self->xmax - self->xmin) * self->w;
	y = self->y + self->h - y / (self->ymax - self->ymin) * self->h;
	self->points[self->npoint - 1].x = x;
	self->points[self->npoint - 1].y = y;
}
void Graph_draw(Graph_area * self,HWND hwnd){
	HDC hdc = GetDC(hwnd);
	POINT esquerdo[] = {self->x, self->y,self->x,self->y + self->h};
	POINT direito[] = {self->x + self->w,self->y,self->x + self->w, self->y + self->h};
	POINT cima[] = {self->x, self->y, self->x + self->w, self->y};
	POINT baixo[] = {self->x, self->y + self->h, self->x + self->w, self->y + self->h};
	Polyline(hdc, esquerdo, 2);
	Polyline(hdc, direito, 2);
	Polyline(hdc, cima, 2);
	Polyline(hdc, baixo, 2);
	Polyline(hdc, self->points, self->npoint);
	ReleaseDC(hwnd,hdc);
}
#endif //GAB_GRAFICO_H
