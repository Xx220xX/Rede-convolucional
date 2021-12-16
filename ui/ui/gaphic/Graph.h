//
// Created by Henrique on 15/12/2021.
//

#ifndef UI_GRAPH_H
#define UI_GRAPH_H

#include "renderizable.h"

typedef struct Graph_t {
	DRAWABLE_CLASS;
	SDL_Texture *texture;
	SDL_Surface *surface;
	SDL_Renderer *renderer;
	SDL_Rect graph_area;
	SDL_Rect title_area;
	SDL_FPoint ylim;
	SDL_FPoint xlim;
	void (*plotF)(struct Graph_t *self,float *x,float *y, int len);
	void (*clear)(struct Graph_t *self);
	void (*setSize)(struct Graph_t *self,int w,int h);
} *Graph;
extern  void runOnUiThread(void(*f)(void *),void *data);
Graph Graph_new(int w, int h, SDL_Renderer *renderer,SDL_Window *window);
#define Map(x,y,len,body){for(int i=0;i<len;i++){body}}
#endif //UI_GRAPH_H
