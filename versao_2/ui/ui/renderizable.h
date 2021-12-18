//
// Created by Henrique on 15/12/2021.
//

#ifndef UI_RENDERIZABLE_H
#define UI_RENDERIZABLE_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_test.h>
#define  DRAWABLE_CLASS void (*render)(void *self, SDL_Renderer *renderer);void (*release)(void *selfp);SDL_Rect  rt;uint8_t focuss
#define  CRENDERIZABLE (void (*)(void *self, SDL_Renderer *renderer))
#define  CREALIZABLE (void (*)(void *self))

typedef struct {
	DRAWABLE_CLASS;
} *DrawableClass;
#endif //UI_RENDERIZABLE_H
