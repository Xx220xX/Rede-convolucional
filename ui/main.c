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
	int showed = 0;
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

	Graph figure1 = Graph_new(500, 500, App.renderer, App.window);
	App.pushImage(figure1);
	int len = 1000;
	float x[len], y[len];
	float ts = 1.0 / len;
	float f = 10;
	figure1->ylim.x = -1;
	figure1->ylim.y = 1;
	figure1->xlim.x = -1;
	figure1->xlim.y = 1;
	Map(x, y, len,
		x[i] = -1 + i * 2.0 / len;
				y[i] = x[i];
	//sin(2 * M_PI * f * x[i]);
	   )
	figure1->plotF(figure1, x, y, len);


//	memset(y,0,sizeof(float)*100);
//	figure1->plotF(figure1, x, y, 100);

	// eventos do sistema, teclado, mouse, etc
	int mouseX, mouseY;
	int relativex, relativey;
	while (App.runing) {
		if (figure1->focuss) {

			SDL_GetMouseState(&mouseX, &mouseY);
			mouseX -= relativex;
			mouseY -= relativey;
			if (!(mouseX < 0 || mouseX > App.w - figure1->rt.w)) {
				figure1->rt.x = mouseX;
			}
			if (!(mouseY < 0 || mouseY > App.h - figure1->rt.h)) {
				figure1->rt.y = mouseY;
			}
		}
		if (SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT:
					App.runing = 0;
					break;
				case SDL_MOUSEBUTTONDOWN:
//					printf("Mouse down %d %d\n", event.button.button, event.button.x);
					SDL_GetMouseState(&mouseX, &mouseY);
					if (!(mouseX < figure1->rt.x || mouseX > figure1->rt.x + figure1->rt.w ||
						  mouseY < figure1->rt.y || mouseY > figure1->rt.y + figure1->rt.h)) {
						relativex = mouseX - figure1->rt.x;
						relativey = mouseY - figure1->rt.y;
						figure1->focuss = 1;
					}
					break;
				case SDL_MOUSEBUTTONUP:
					if (figure1->focuss) { figure1->focuss = 0; }
//					printf("Mouse up\n");
					break;
				case SDL_KEYDOWN:
					switch (event.key.keysym.sym) {
						case SDLK_1: {

							f= (rand() / (float) RAND_MAX) * 6;
							Map(x, y, len, y[i] = sin(2*M_PI*f*x[i]);)
							figure1->clear(figure1);
							figure1->plotF(figure1, x, y, len);
						}
							break;
						default:
							printf("%d %c %lc\n", event.key.keysym.sym, event.key.keysym.sym, (wchar_t) event.key.keysym.sym);
							break;
					}
			}
		}
	}

// cleanup SDL

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