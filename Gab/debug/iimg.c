//
// Created by Henrique on 30/11/2021.
//

#include<string.h>
#include <stdarg.h>
#include <png/png.h>

#include "cnn/cnn.h"


void Cnn_asimage(Cnn c,char *file, int largura, int altura, ...) {
	Tensor s = NULL;
	va_list v;
	uint8_t *image = alloc_mem(altura, largura);
	va_start(v, altura);
	s = va_arg(v, Tensor);
	int nTensors = 0;
	while (s) {
		nTensors++;
		s = va_arg(v, Tensor);
	}
	va_end(v);
	va_start(v, altura);
	int w = largura / nTensors;
	int h;
	int k;
	for (int j = 0; j < nTensors; ++j) {
		s = va_arg(v, Tensor);
		h = altura / (s->z * s->w);
		for (int l = 0; l < s->w; ++l) {
			for (int z = 0; z < s->z; ++z) {
				k = w*s->z + z;
				s->imagegray(s, image, largura, altura, w, h, k*h, j * w, z, l);
			}
		}
	}
	va_end(v);
	png(file,image,largura,altura);
	free_mem(image);
}

int main() {
	Cnn cnn = Cnn_new();

	return cnn->release(&cnn);
}