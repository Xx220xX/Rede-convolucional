#ifndef UI_PLOT_H
#define UI_PLOT_H

#include <math.h>

#define  WPAD 30
#define  HPAD 30
//#define  pfunc printf("%s\n",__FUNCTION__);
#define  pfunc
typedef struct {
	void *fig;
	POINTFLOAT *pf;
	int size;
	DWORD lineColor;

	void (*push)(void *self, float x, float y);

	void (*pushDraw)(void *self, float x, float y);

	void (*draw)(void *self, HDC hdc);

	void (*release)(void *self);
} Axe;
typedef struct {
	HWND window;
	RECT winrc;//retangulo da janela
	RECT grc;// retangulo do grafico

	float xmin, xmax;
	float ymin, ymax;
	int wpad;
	int hpad;
	char *xlabel;
	char *ylabel;
	int grid;
	char gformat[10];
	float xstep;
	float ystep;
	float ygrid_start;
	float xgrid_start;
	char *title;
	int nstepx;
	int nstepy;
	DWORD bkcolor;
	Axe *axes;

	size_t naxes;

	void (*putAxe)(void *self, DWORD lineColor);

	Axe *(*getAxe)(void *self, int id);

	void (*release)(void *self);

	void (*draw)(void *self, HDC hdc);

	void (*setVisible)(void *self, int visible);


} Figure;

LRESULT CALLBACK Figure_Proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	pfunc
	Figure *self;
	switch (msg) {
		case WM_CREATE:

			break;
		case WM_CLOSE:
			break;
		case WM_MOVE:
		case WM_UPDATEUISTATE:
		case WM_PAINT: {
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hwnd, &ps);
			FillRect(hdc, &ps.rcPaint, (HBRUSH) (COLOR_WINDOW));
			self = (Figure *) GetWindowLongPtrA(hwnd, 0);
			if (self) {
				self->winrc = ps.rcPaint;
				self->grc.left = self->winrc.left + self->wpad;
				self->grc.right = self->winrc.right - self->wpad;
				self->grc.top = self->winrc.top + self->hpad;
				self->grc.bottom = self->winrc.bottom - self->winrc.top - self->hpad;
				self->draw(self, hdc);
			}
			// All painting occurs here, between BeginPaint and EndPaint.
			EndPaint(hwnd, &ps);
		}
			break;
		case WM_DESTROY:
			break;
//		case WM_MOVE:
		case WM_SIZE: {
			RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);
			break;
		}
	}
	return DefWindowProcW(hwnd, msg, wParam, lParam);
}

void Axe_release(Axe *self) {
	pfunc
	if (self->pf) {
		free(self->pf);
	}
	self->pf = NULL;
	self->size = 0;
}

#define between(x, mx, mn)x>+mx?mx:(x<=mn?mn:x)

POINT Figure_convert(Figure *self, float x, float y) {
	RECT rect = self->grc;
	x = between(x, self->xmax, self->xmin);
	y = between(y, self->ymax, self->ymin);
	POINT p;
	p.x = rect.left + (x - self->xmin) / (self->xmax - self->xmin) * (rect.right - rect.left);
	p.y = rect.bottom + (y - self->ymin) / -(self->ymax - self->ymin) * (rect.bottom - rect.top);
	return p;
};

POINT Axe_convert(Axe *self, float x, float y) {
	pfunc
//	Figure *figure = self->fig;
//	RECT rect = figure->grc;
//	x = between(x, figure->xmax, figure->xmin);
//	y = between(y, figure->ymax, figure->ymin);
//	POINT p;
//	p.x = rect.left + (x - figure->xmin) / (figure->xmax - figure->xmin) * rect.right;
//	p.y = (y + figure->ymin) / (figure->ymin - figure->ymax) * (rect.bottom - rect.top) + rect.top;
	return Figure_convert(self->fig, x, y);
}

#define MAXLEN 1000

void Axe_push(Axe *self, float x, float y) {
	pfunc
	self->size++;
	self->pf = realloc(self->pf, self->size * sizeof(POINTFLOAT));
	self->pf[self->size - 1].x = x;
	self->pf[self->size - 1].y = y;
	if (self->size > MAXLEN) {
		double px = (double) self->size / MAXLEN;
		int j;
		POINTFLOAT *f = calloc(MAXLEN, sizeof(POINTFLOAT));
		for (int i = 0; i < MAXLEN; ++i) {
			j = i * px +0.5;
			f[i] = self->pf[j];
		}
		free(self->pf);
		self->pf = f;
		self->size = MAXLEN;
//		((Figure *)self->fig)->draw(self->fig,NULL);
	}

}


void Axe_pusDraw(Axe *self, float x, float y) {
	pfunc
	Axe_push(self, x, y);
	Figure *figure = self->fig;
	if (self->size < 2) {
		return;
	}
	HDC hdc = GetDC(figure->window);
	HPEN hPen = CreatePen(PS_JOIN_ROUND, 1, self->lineColor);
	SelectObject(hdc, hPen);
	int i = self->size - 1;
	POINT p[2];
	if (!(self->pf[i].x < figure->xmin && self->pf[i - 1].x < figure->xmin || self->pf[i].x > figure->xmax && self->pf[i - 1].x > figure->xmax || self->pf[i].y < figure->ymin && self->pf[i - 1].y < figure->ymin || self->pf[i].y > figure->ymax && self->pf[i - 1].y > figure->ymax)) {
		p[0] = Axe_convert(self, self->pf[i - 1].x, self->pf[i - 1].y);
		p[1] = Axe_convert(self, self->pf[i].x, self->pf[i].y);
		Polyline(hdc, p, 2);

	}
	DeleteObject(hPen);
	ReleaseDC(figure->window, hdc);
}

void Axe_draw(Axe *self, HDC hdc) {
	pfunc
	if (self->size <= 0) {
		return;
	}
	int releasehdc = 0;
	Figure *figure = self->fig;
	if (!hdc) {
		hdc = GetDC(figure->window);
		releasehdc = 1;
	}
	HPEN hPen = CreatePen(PS_JOIN_ROUND, 1, self->lineColor);
	SelectObject(hdc, hPen);
	POINT p[2];
	for (int i = 1; i < self->size; ++i) {
		if (!(self->pf[i].x < figure->xmin && self->pf[i - 1].x < figure->xmin || self->pf[i].x > figure->xmax && self->pf[i - 1].x > figure->xmax || self->pf[i].y < figure->ymin && self->pf[i - 1].y < figure->ymin || self->pf[i].y > figure->ymax && self->pf[i - 1].y > figure->ymax)) {
			p[0] = Axe_convert(self, self->pf[i - 1].x, self->pf[i - 1].y);
			p[1] = Axe_convert(self, self->pf[i].x, self->pf[i].y);
			Polyline(hdc, p, 2);
		}
	}
	DeleteObject(hPen);
	if (releasehdc) {
		ReleaseDC(figure->window, hdc);
	}
}

int Figure_INITW(HINSTANCE hInsta) {
	pfunc
	WNDCLASSEX wc;
	wc.hInstance = hInsta; // inj_hModule;
	wc.lpszClassName = "Figure";
	wc.lpfnWndProc = Figure_Proc;
	wc.style = CS_DBLCLKS;
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.lpszMenuName = NULL;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = sizeof(Figure *);
	wc.hbrBackground = (HBRUSH) WHITE_BRUSH;
	return RegisterClassEx(&wc);
}

void Figure_grid(Figure *self, HDC hdc) {
	HPEN hPen = CreatePen(PS_JOIN_ROUND, 1, RGB(122, 122, 122));
	SelectObject(hdc, hPen);
	POINT p[2];
	double x = self->xgrid_start;
	double y = self->ygrid_start;
	if (isnan(y)) {
		y = self->ymin;
	}
	if (isnan(x)) {
		x = self->xmin;
	}
	char buff[15];
	int len;
	SIZE ft;
	float step = self->xstep;
	if (self->nstepx != 0) {
		step = (self->xmax - self->xmin) / self->nstepx;
	}
	do {
		if (x >= self->xmin) {
			p[0] = Figure_convert(self, x, self->ymin);
			p[1] = Figure_convert(self, x, self->ymax);
			len = snprintf(buff, 15, self->gformat, x);
			GetTextExtentPoint32A(hdc, buff, len, &ft);
			TextOutA(hdc, p[0].x - ft.cx / 2, p[0].y + ft.cy / 4, buff, len);
			Polyline(hdc, p, 2);
		}
		x = x + step;
	} while (x <= self->xmax);
	step = self->ystep;
	if (self->nstepy != 0) {
		step = (self->ymax - self->ymin) / self->nstepy;
	}
	do {
		if (y >= self->ymin) {
			p[0] = Figure_convert(self, self->xmin, y);
			p[1] = Figure_convert(self, self->xmax, y);
			len = snprintf(buff, 15, self->gformat, y);
			GetTextExtentPoint32A(hdc, buff, len, &ft);
			TextOutA(hdc, p[0].x - (1 + 1.0 / len) * ft.cx, p[0].y - ft.cy / 2, buff, len);
			Polyline(hdc, p, 2);
		}
		y = y + step;
	} while (y <= self->ymax);
	DeleteObject(hPen);
}

void Figure_draw(Figure *self, HDC hdc) {
	pfunc
	int releasehdc = 0;
	if (!hdc) {
		hdc = GetDC(self->window);
		releasehdc = 1;
	}
	RECT r = self->grc;

	HBRUSH brush = CreateSolidBrush(self->bkcolor);
	FillRect(hdc, &r, brush);
	DeleteObject(brush);

	if (self->title) {
		SIZE ft;
		GetTextExtentPoint32A(hdc, self->title, strlen(self->title), &ft);
		TextOutA(hdc, (self->winrc.right + self->winrc.left) / 2 - ft.cx / 2, 0, self->title, strlen(self->title));
	}
	if (self->grid) {
		Figure_grid(self, hdc);
	}
	Axe *axe;
	for (int i = 0; i < self->naxes; ++i) {


		axe = self->getAxe(self, i);
		axe->draw(axe, hdc);

	}
	if (releasehdc) {
		ReleaseDC(self->window, hdc);
	}

}

void Figure_release(Figure *self) {
	pfunc
	DestroyWindow(self->window);
	if (self->axes) {
		for (int i = 0; i < self->naxes; ++i) {
			self->axes[i].release(self->axes + i);
		}
		free(self->axes);
	}
	memset(self, 0, sizeof(Figure));
}

void Figure_putAxe(Figure *self, DWORD lineColor) {
	pfunc
	self->axes = realloc(self->axes, (self->naxes + 1) * sizeof(Axe));
	self->axes[self->naxes] = (Axe) {0};
	self->axes[self->naxes].fig = self;
	self->axes[self->naxes].lineColor = lineColor;
	self->axes[self->naxes].release = (void (*)(void *)) Axe_release;
	self->axes[self->naxes].push = (void (*)(void *, float, float)) Axe_push;
	self->axes[self->naxes].draw = (void (*)(void *, HDC)) Axe_draw;
	self->axes[self->naxes].pushDraw = (void (*)(void *, float, float)) Axe_pusDraw;
	self->naxes++;
}

Axe *Figure_getAxe(Figure *self, int id) {
	pfunc
	if (id < 0 || id >= self->naxes) {
		return NULL;
	}
	return &self->axes[id];
}

void Figure_setVisible(Figure *self, int visible) {
	if (!visible) {
		ShowWindow(self->window, SW_HIDE);
	} else {
		ShowWindow(self->window, SW_SHOWDEFAULT);
	}
	RedrawWindow(GetParent(self->window), NULL, NULL, RDW_INVALIDATE | RDW_UPDATENOW);

}

int Figure_new(Figure *self, HWND parente, HINSTANCE hInstance, int x, int y, int w, int h) {
	pfunc
	*self = (Figure) {0};
	snprintf(self->gformat, 10, "%s", "%.1lf");
	self->wpad = 30;
	self->hpad = 30;
	self->xgrid_start = NAN;
	self->ygrid_start = NAN;
	self->xmin = self->ymin = -1;
	self->xmax = self->ymax = 1;
	self->draw = (void (*)(void *, HDC)) Figure_draw;
	self->release = (void (*)(void *)) Figure_release;
	self->putAxe = (void (*)(void *, DWORD)) Figure_putAxe;
	self->getAxe = (Axe *(*)(void *, int)) Figure_getAxe;
	self->setVisible = (void (*)(void *, int)) Figure_setVisible;
	self->winrc = (RECT) {.left = x, .right = w, .top = y, .bottom = h};
	self->title = "teste";
	self->window = CreateWindowW(L"Figure", NULL, WS_CHILD | WS_CLIPSIBLINGS | WS_VISIBLE, x, y, w, h, parente, NULL, hInstance, self);
	SetWindowLongPtrA(self->window, 0, (LONG_PTR) self);
	return 0;
}

#endif //UI_PLOT_H
