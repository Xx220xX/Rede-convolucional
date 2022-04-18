//
// Created by Henrique on 13/04/2022.
//

#ifndef MAIN_C_IMAGE_H
#define MAIN_C_IMAGE_H
typedef struct {
	unsigned char *raw;
	int w, h;
	int channels;
} Image;

void ReleaseImage(Image *image) {
	if (image->raw) {
		free(image->raw);
	}
	*image = (Image) {0};
}

int loadPPM(char *file_name, Image *im) {
	ReleaseImage(im);
	int error = 0;
	char buff[500];
	int w = 0, h = 0;
	int max = 0;
	FILE *f = fopen(file_name, "r");
	if (!f) {
		error = -1;
		goto end;
	}
	char type = fgetc(f);
	if (type != 'P') {
		error = -2;
		goto end;
	}
	type = fgetc(f);
	int c = 0;
	// procurar pela largura
	while (!feof(f)) {
		c = fgetc(f);
		if (c == ' ' || c == '\n' || c == '\t') {
			// achou
			break;
		}
		if (c == '#') {// é um comentário
			while (!feof(f) && fgetc(f) != '\n');
			continue;
		}
		// tem que ser um numero
		c = fgetc(f);
		if (c < '0' || c > '9') {
			error = -4;
			goto end;
		}
		w = w * 10 + c - '0';
	}
	// procurar pela altura
	while (!feof(f)) {
		c = fgetc(f);
		if (c == ' ' || c == '\n' || c == '\t') {
			// achou
			break;
		}
		if (c == '#') {// é um comentário
			while (!feof(f) && fgetc(f) != '\n');
			continue;
		}
		// tem que ser um numero
		c = fgetc(f);
		if (c < '0' || c > '9') {
			error = -4;
			goto end;
		}
		h = h * 10 + c - '0';
	}
// procurar pelo maximo
	while (!feof(f)) {
		c = fgetc(f);
		if (c == ' ' || c == '\n' || c == '\t') {
			// achou
			break;
		}
		if (c == '#') {// é um comentário
			while (!feof(f) && fgetc(f) != '\n');
			continue;
		}
		// tem que ser um numero
		c = fgetc(f);
		if (c < '0' || c > '9') {
			error = -4;
			goto end;
		}
		max = max * 10 + c - '0';
	}

	int n;
	switch (type) {
		case '2':
			im->raw = calloc(h, w);
			for (int i = 0; i < h; ++i) {
				for (int j = 0; j < w; ++j) {
					n = 0;
					while (!feof(f)) {
						if (c == ' ' || c == '\n') {
							break;
						}
						n = n * 10 + c;
					}
					if (max != 255) {
						n = n * 255.0 / max;
					}
					im->raw[i * w + j] = n;
				}
			}
			im->channels = 1;
			break;
		case '3':
			im->raw = calloc(3 * h, w);
			for (int i = 0; i < h * 3; ++i) {
				for (int j = 0; j < w; ++j) {
					n = 0;
					while (!feof(f)) {
						if (c == ' ' || c == '\n') {
							break;
						}
						n = n * 10 + c;
					}
					if (max != 255) {
						n = n * 255.0 / max;
					}
					im->raw[i * w + j] = n;
				}
			}
			im->channels = 3;
			break;
		case '5':
			im->raw = calloc(h, w);
			fread(im->raw, w, h, f);
			im->channels = 1;
			break;
		case '6':
			im->raw = calloc(3 * h, w);
			fread(im->raw, w, 3 * h, f);
			im->channels = 3;
			break;

	}
	im->w = w;
	im->h = h;
	end:
	if (f) {
		fclose(f);
	}
	return error;
}

#endif //MAIN_C_IMAGE_H
