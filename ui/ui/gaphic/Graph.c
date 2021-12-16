//
// Created by Henrique on 15/12/2021.
//

#include <stdio.h>
#include "Graph.h"
#include "float.h"

#define min_max_vec(minimo, maximo, v, len)do{    \
minimo = (typeof(minimo))DBL_MAX;\
maximo = (typeof(minimo))DBL_MIN;\
for(int i=0;i<len;i++){\
    if(maximo<v[i]){maximo = v[i];}\
    if(minimo>v[i]){minimo = v[i];}\
}}while(0)

void Graph_release(Graph *selfp) {
	if (!selfp || !*selfp) { return; }
	SDL_DestroyTexture((*selfp)->texture);
	SDL_DestroyRenderer((*selfp)->renderer);
	free(*selfp);
	*selfp = NULL;
}

void Graph_render(Graph self, SDL_Renderer *renderer) {
	SDL_RenderCopy(renderer, self->texture, NULL, &self->rt);
}

void _Graph_clear(Graph self) {
	SDL_SetRenderTarget(self->renderer, self->texture);
	SDL_SetRenderDrawColor(self->renderer, 0xff, 0xff, 0xff, 0xff);
	SDL_RenderClear(self->renderer);
	SDL_SetRenderTarget(self->renderer, NULL);
	SDL_SetRenderDrawColor(self->renderer, 0, 0, 0, 1);
}

#define  PB printf("%d\n",
#define PE )

void _Graph_Draw_lines(void **data) {
	static int n = 1;
	printf("call %d\n", n++);
	Graph self = data[0];
	SDL_Point *pontos = data[1];
	int len = (int) (int64_t) data[2];
	PB SDL_RenderFlush(self->renderer)PE;
	PB SDL_SetRenderTarget(self->renderer, self->texture)PE;
	PB SDL_RenderDrawLines(self->renderer, pontos, len)PE;
	PB SDL_RenderFlush(self->renderer) PE;
	PB SDL_SetRenderTarget(self->renderer, NULL)PE;
//	for (int i = 0; i < len; ++i) {
//		printf("%d %d %d %d\n", pontos[i].x, pontos[i].y, self->graph_area.w, self->graph_area.h);
//
//	}

	free(data);
	free(pontos);
}

void *makeData(Graph self, ...) {
	void **mem = calloc(1, sizeof(void *));
	mem[0] = self;
	int size_mem = 1;
	va_list v;
	va_start(v, self);
	int len;
	void *buff;
	for (void *p = va_arg(v, void *); p; p = va_arg(v, void *)) {
		len = va_arg(v, int);
		if (len == -1) {
			buff = p;
		} else {
			buff = calloc(1, len);
			memcpy(buff, p, len);
		}
		size_mem++;
		mem = realloc(mem, (size_mem) * sizeof(void *));
		mem[size_mem - 1] = buff;
	}
	va_end(v);
	return mem;
}

void _Graph_Draw_2_points(void **data) {
	Graph self = data[0];
	SDL_Point pontos[2] = {*((SDL_Point *) data[1]), *((SDL_Point *) data[2])};
	free(data[1]);
	free(data[2]);
	free(data);
	SDL_SetRenderTarget(self->renderer, self->texture);
	SDL_RenderDrawLines(self->renderer, pontos, 2);
	SDL_SetRenderTarget(self->renderer, NULL);
}


void Graph_xticks(Graph self) {
	SDL_Point p0, p1;
	p0.x = 0;
	p0.y = self->graph_area.h + 1;
	p1.x = self->rt.w;
	p1.y = self->graph_area.h + 1;
	runOnUiThread((void (*)(void *)) _Graph_Draw_2_points, makeData(self, &p0, sizeof(SDL_Point), &p1, sizeof(SDL_Point), NULL));
	p0.x = self->graph_area.x - 1;
	p0.y = 0;
	p1.x = self->graph_area.x - 1;
	runOnUiThread((void (*)(void *)) _Graph_Draw_2_points, makeData(self, &p0, sizeof(SDL_Point), &p1, sizeof(SDL_Point), NULL));
	p1.y = self->rt.h;
}

#pragma GCC push_options
#pragma GCC optimize ("O0")

void Graph_plotF(Graph self, float *x, float *y, int len) {
	if (len <= 0) { return; }
	SDL_Point *pontos = calloc(len, sizeof(SDL_Point));
	SDL_FPoint xlim = self->xlim, ylim = self->ylim;
	if (xlim.x == xlim.y) {
		min_max_vec(xlim.x, xlim.y, x, len);
	}
	if (ylim.x == ylim.y) {
		min_max_vec(ylim.x, ylim.y, y, len);
	}
	if (xlim.x == xlim.y) {
		xlim.x -= 1;
		xlim.y += 1;
	}
	if (ylim.x == ylim.y) {
		ylim.x -= 1;
		ylim.y += 1;
	}
	self->setSize(self, self->rt.w, self->rt.h);
	float pw = ((float) self->graph_area.w) / (xlim.y - xlim.x);
	float ph = ((float) self->graph_area.h) / (ylim.y - ylim.x);

	for (int i = 0; i < len; ++i) {
		pontos[i].x = self->graph_area.x + (x[i] - xlim.x) * pw;
		pontos[i].y = self->graph_area.h - (y[i] - ylim.x) * ph;
//		printf("%d %d %d %d\n", pontos[i].x, pontos[i].y,self->graph_area.w,self->graph_area.h);
	}

	void **data = calloc(3, sizeof(void *));
//	printf("here %d %d\n",len,pontos[0].x);
	data[0] = self;
	data[1] = pontos;
	data[2] = (void *) (int64_t) len;
	runOnUiThread((void (*)(void *)) _Graph_Draw_lines, data);
	Graph_xticks(self);

}

#pragma GCC pop_options

void Graph_clear(Graph self) {
	runOnUiThread((void (*)(void *)) _Graph_clear, self);
}

void Grap_setSize(Graph self, int w, int h) {
	self->rt.w = w;
	self->rt.h = h;
	self->graph_area.w = w * 0.9;
	self->graph_area.h = h * 0.9;
	self->graph_area.x = w * 0.1;
	self->graph_area.y = h * 0.1;
}

Graph Graph_new(int w, int h, SDL_Renderer *renderer, SDL_Window *window) {
	Graph self = calloc(1, sizeof(struct Graph_t));
	Grap_setSize(self, w, h);
	self->rt.x = 0;
	self->rt.y = 0;
	self->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET, w, h);
	self->renderer = renderer;
	if (!self->texture || !self->renderer) {
		fprintf(stderr, "ERRO %s\n", SDL_GetError());
		exit(-100);
	}

	self->release = CREALIZABLE Graph_release;
	self->render = CRENDERIZABLE Graph_render;
	self->plotF = Graph_plotF;
	self->clear = Graph_clear;
	self->setSize = Grap_setSize;
	self->clear(self);
	return self;
}
