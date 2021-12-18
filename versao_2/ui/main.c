#define SDL_MAIN_HANDLED

#include <SDL2/SDL.h>
#include <SDL2/SDL_timer.h>
#include <stdio.h>
#include <time.h>
#include <gaphic/Graph.h>
#include <math.h>
#include "imagem/Imagem.h"
#include "renderizable.h"
#include "list/list.h"

struct {
	SDL_Window *window;
	SDL_Renderer *renderer;
	Lista toDraw;
	Lista runFunctions;
	uint8_t runing;
	int w;
	int h;
	double setFps;
	double getFps;

	void (*pushImage)(void *render);
} App = {0, .setFps = 60};

void App_pushRender(void *renderizable) {
	App.toDraw.push(&App.toDraw, renderizable);
}

void runOnUiThread(void(*f)(void *), void *data) {
	void **tmp = calloc(2, sizeof(void *));
	tmp[0] = f;
	tmp[1] = data;
	App.runFunctions.push(&App.runFunctions, tmp);
}

#include "windows.h"
double SDL_seconds() {
	FILETIME ft;
	LARGE_INTEGER li;
	GetSystemTimePreciseAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	u_int64 ret = li.QuadPart;
	return (double) ret * 1e-7;
}
int draw(void *unused) {
	void **tmp;
	void (*f)(void *);
	void *data;
	double a, b = SDL_seconds(), delta;
	double target = 1.0 / App.setFps;
	double dta = target;
	while (App.runing) {
		a = SDL_seconds();
		delta = a - b;
		if (delta > dta) {
			App.toDraw.lock(&App.toDraw);
			App.getFps = 1.0 / delta;
			b = a;
			dta = dta - 1e-4 * (delta - target);
			SDL_RenderClear(App.renderer);
			for (int i = 0; i < App.toDraw.length; ++i) {
				((DrawableClass) App.toDraw.elements[i])->render(App.toDraw.elements[i], App.renderer);
//				printf("%d %d\n",img->rt.w,img->rt.h);
			}
			SDL_RenderPresent(App.renderer);
			SDL_RenderFlush(App.renderer);
			App.toDraw.unlock(&App.toDraw);

		}
		if (App.runFunctions.length > 0) {
			tmp = App.runFunctions.pop(&App.runFunctions);
			f = tmp[0];
			data = tmp[1];
			free(tmp);
			f(data);
		}
		Sleep(0);
	}
	printf("exit ok draw\n");
	return 0;
}

#define ICONS_PATH "../icons/"


int main(int argc, char **argv) {

	srand(time(0));
// variables
	SDL_Event event;
	App.pushImage = App_pushRender;
	App.toDraw = Lista_new();
	App.runFunctions = Lista_new();
	SDL_Init(SDL_INIT_VIDEO);
	App.w = 800;
	App.h = 600;
	App.window = SDL_CreateWindow("SDL2 Keyboard/Mouse events", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, App.w, App.h, 0);
	App.renderer = SDL_CreateRenderer(App.window, -1, 0);
	SDL_SetRenderDrawColor(App.renderer, 0, 0, 0, 1);
	App.runing = 1;
	// começa a desenhar na tela
	SDL_Thread *handle_draw = SDL_CreateThread(draw, "thread_draw", NULL);

//	, mouse, etc
	int mouseX, mouseY;
	int relativex, relativey;
	while (App.runing) {

		if (SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT:
					App.runing = 0;
					break;
				case SDL_MOUSEBUTTONDOWN:

					break;
				case SDL_MOUSEBUTTONUP:

					break;
				case SDL_KEYDOWN:
					break;

			}
		}


	}

//	SDL_DestroyTexture(canvas);
	int status = 0;
	App.runing = 0;// flag para encerrar
	SDL_WaitThread(handle_draw, &status);// esperar a thread draw encerrar
	SDL_DetachThread(handle_draw); // exclui a thread draw
	DrawableClass drawableClass;
	while (App.toDraw.length > 0) {
		drawableClass = App.toDraw.pop(&App.toDraw);
		drawableClass->release(&drawableClass);
	}
	SDL_DestroyRenderer(App.renderer);// exclui o renderer
	SDL_DestroyWindow(App.window);// exclui a janelaa
	SDL_Quit(); // encerra a api sdl
	return 0;
}
