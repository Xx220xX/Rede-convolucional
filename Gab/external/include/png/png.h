#ifndef PNG_PNG_H
#define PNG_PNG_H

#include <stdio.h>

extern unsigned char *rescale(unsigned char *img, unsigned int h0, unsigned int w0, unsigned int h, unsigned int w);

extern unsigned char *rescaleRGB(unsigned char *img, unsigned int h0, unsigned int w0, unsigned int h, unsigned int w);

extern int pngGRAY(const char *filename, const unsigned char *image, unsigned width, unsigned height);
extern int pngGRAYF(FILE *f, const unsigned char *image, unsigned width, unsigned height);

extern int png(const char *filename, const unsigned char *image, unsigned width, unsigned height);

extern int pngRGB(const char *filename, const unsigned char *red, const unsigned char *green, const unsigned char *blue, unsigned width, unsigned height);

#endif //PNG_PNG_H
