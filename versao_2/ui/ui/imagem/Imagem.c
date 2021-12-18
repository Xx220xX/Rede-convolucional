//
// Created by Henrique on 13/12/2021.
//

#include "Imagem.h"
#include "stdio.h"

SDL_Surface *creategrayImg(int w, int h, unsigned char *red, unsigned char *green, unsigned char *blue) {
	int *rgb = calloc(w * h, sizeof(int));
	for (int i = w * h - 1; i >= 0; --i) {
		rgb[i] = myRGB(red[i], green[i], blue[i]);
	}
	SDL_Surface *s = SDL_CreateRGBSurfaceFrom(rgb, w, h, 32, 4 * w, _R, _G, _B, _A);
	free(rgb);
	return s;
}

SDL_Surface *createImg() {
	unsigned char img[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18,
						   126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						   0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	return creategrayImg(28, 28, img, img, img);
}

void Img_setImaage(Img img, uint8_t *red, uint8_t *green, uint8_t *blue, uint8_t *alpha) {
//	Uint32* pixels = nullptr;
	int pitch = 0;
	uint32_t format = SDL_PIXELFORMAT_RGBA32;
	uint32_t *pixels;

// Get the size of the texture.
	int w, h;
	SDL_QueryTexture(img->texture, &format, NULL, &w, &h);

// Now let's make our "pixels" pointer point to the texture data.
	if (SDL_LockTexture(img->texture, NULL, (void **) &pixels, &pitch)) {
		// If the locking fails, you might want to handle it somehow. SDL_GetError(); or something here.
		fprintf(stderr, "Falha %s\n", SDL_GetError());
		return;
	}
	SDL_PixelFormat pixelFormat = {0};
	pixelFormat.format = format;


// Now you want to format the color to a correct format that SDL can use.
// Basically we convert our RGB color to a hex-like BGR color.
//	Uint32 color = SDL_MapRGB(&pixelFormat, R, G, B);
// Before setting the color, we need to know where we have to place it.
	Uint32 pixelPosition;
	unsigned int ww = pitch / sizeof(unsigned int);
	for (int y = 0; y < ww; ++y) {
		for (int x = 0; x < ww; ++x) {
			pixelPosition = y * (ww) + x;
			uint32_t cor = SDL_MapRGBA(&pixelFormat, red[y * w + x], green ? green[y * w + x] : 0, blue ? blue[y * w + x] : 0, alpha ? alpha[y * w + x] : 0xff);
			pixels[pixelPosition] = cor;

//			pixels[pixelPosition] = SDL_MapRGBA(&pixelFormat, red ? red[y * w + x] : 0, green ? green[y * w + x] : 0, blue ? blue[y * w + x] : 0, alpha ? alpha[y * w + x] : 0xff);
		}
	}
// Now we can set the pixel(s) we want.
// Also don't forget to unlock your texture once you're done.
	SDL_UnlockTexture(img->texture);
}

void Img_release(Img *selfp) {
	if (!selfp || !*selfp) { return; }
	SDL_DestroyTexture((*selfp)->texture);
	free(*selfp);
	*selfp = 0;
}

void Img_render(Img self, SDL_Renderer *renderer) {
	SDL_RenderCopy(renderer, self->texture, NULL, &self->rt);
}

Img Img_new(int w, int h, SDL_Renderer *renderer) {
	Img self = calloc(1, sizeof(struct Img_t));
	self->w = self->rt.w = w;
	self->h = self->rt.h = h;
	self->rt.x = 100;
	self->rt.y = 100;
	self->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, w, h);
	if (!self->texture) { exit(-100);}
	self->setPixels = Img_setImaage;
	self->release = CREALIZABLE Img_release;
	self->render = CRENDERIZABLE Img_render;
	return self;
}
