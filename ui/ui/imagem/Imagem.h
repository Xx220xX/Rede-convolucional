//
// Created by Henrique on 13/12/2021.
//

#ifndef UI_IMAGEM_H
#define UI_IMAGEM_H

#include <SDL2/SDL_image.h>

#include "renderizable.h"

#define _A 0xFF000000
#define _R 0x000000FF
#define _G 0x0000FF00
#define _B 0x00FF0000
#define myRGBA(r, g, b, a)(_A&(a<<24)|(r&_R)|((g<<8)&_G)|((b<<16)&_B))
#define myRGB(r, g, b)myRGBA(r,g,b,0xff)

typedef struct Img_t {
	DRAWABLE_CLASS;
	SDL_Texture *texture;
	int w, h;
	void (*setPixels)(struct Img_t *self, uint8_t *red, uint8_t *green, uint8_t *blue, uint8_t *alpha);
} *Img;

Img Img_new(int w, int h, SDL_Renderer *renderer);

#endif //UI_IMAGEM_H
