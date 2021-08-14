//
// Created by Henrique on 28-Jul-21.
//

#include "utils/vectorUtils.h"
#include <windows.h>

void ppmp2(double *data, int x, int y, char *fileName) {
	FILE *f = fopen(fileName, "w");
	fprintf(f, "P2\n");
	fprintf(f, "%d %d\n", y, x);
	fprintf(f, "255\n");
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; ++j) {
			fprintf(f, "%d", (int) (data[i * y + j] * 255));
			if (j < y - 1)fprintf(f, " ");
		}
		if (i < x - 1)
			fprintf(f, "\n");
	}
	fclose(f);
}

void ppmp3(double *data, int x, int y, int z, const char *fileName) {
	FILE *f = fopen(fileName, "w");
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", y, x);
	fprintf(f, "255\n");
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; ++j) {
			for (int k = 0; k < z; ++k) {
				fprintf(f, "%d", (int) (data[k * y * x + i * y + j]));
				if (k < z - 1)fprintf(f, " ");
			}
			if (j < y - 1)fprintf(f, " ");
		}
		if (i < x - 1)
			fprintf(f, "\n");
	}
	fclose(f);
}


void createDir(char *dir) {
	printf("%s\n", dir);
	char msg[250] = {0};
	DWORD dwatt = GetFileAttributes(dir);
	if (dwatt != INVALID_FILE_ATTRIBUTES && dwatt & FILE_ATTRIBUTE_DIRECTORY) {
		sprintf(msg, "rmdir /S /Q %s", dir);
		system(msg);
	}
	sprintf(msg, "mkdir  %s", dir);
	system(msg);
}

void salveCnnOutAsPPM(Cnn c, const char *name) {
	size_t w = 0, h = 0;

	char *im = salveCnnOutAsPPMGPU(c, &h, &w);
	FILE *f = fopen(name, "wb");
	fprintf(f, "P5 ");
	fprintf(f, "%zu %zu ", w, h);
	fprintf(f, "255 ");
	fwrite(im, h, w, f);
	free_mem(im);
	fclose(f);
}

void salveTensorAsPPM(const char *name, Tensor t, Cnn c) {
	double *dt = alloc_mem(t->bytes, 1);
	FILE *f = fopen(name, "w");
	TensorGetValues(c->queue, t, dt);
	normalizeGPU(c, dt, dt, t->bytes / sizeof(double), 255, 0);
	fprintf(f, "P2\n");
	fprintf(f, "%d %d\n", t->y * t->z + t->z - 1, t->x);
	fprintf(f, "255\n");
	int px;
	for (int i = 0; i < t->x; i++) {
		for (int z = 0; z < t->z; ++z) {
			for (int j = 0; j < t->y; ++j) {
				px = (int) (dt[z * t->x * t->y + i * t->y + j]);
				fprintf(f, "%d", px & 0xff);
				if (j < t->y - 1)fprintf(f, " ");
			}
			if (z < t->z - 1)fprintf(f, " 0 ");
		}
		if (i < t->x - 1)fprintf(f, "\n");
	}

	free_mem(dt);
	fclose(f);
}

int salveTensorAsPPM3(const char *name, Tensor t, Cnn c) {
	return salveTensor4DAsPPM3(name, t, c, 0);

}

int salveTensor4DAsPPM3(const char *name, Tensor t, Cnn c, UINT w) {
	if (t->z != 3)return -1;
	double *dt = alloc_mem(t->bytes, 1);
	TensorGetValuesOffSet(c->queue, t, dt, t->bytes * w);
	normalizeGPU(c, dt, dt, t->bytes / sizeof(double), 255, 0);
	ppmp3(dt, t->x, t->y, t->z, name);
	free_mem(dt);
	return 0;
}

int salveTensor4DAsPPM(const char *name, Tensor t, Cnn c, UINT w) {
	if (w >= t->w) return -2;
	double *dt = alloc_mem(t->bytes, 1);
	c->error.error = TensorGetValuesOffSet(c->queue, t, dt, t->bytes * w);
	if (c->error.error) {
		free_mem(dt);
		return c->error.error;
	}
	normalizeGPU(c, dt, dt, t->bytes / sizeof(double), 255, 0);
	if (c->error.error) {
		free_mem(dt);
		return c->error.error;
	}

	FILE *f = fopen(name, "w");
	if (!f)return -1;

	fprintf(f, "P2\n");
	fprintf(f, "%d %d\n", t->y * t->z + t->z - 1, t->x);
	fprintf(f, "255\n");
	int px;
	for (int i = 0; i < t->x; i++) {
		for (int z = 0; z < t->z; ++z) {
			for (int j = 0; j < t->y; ++j) {
				px = (int) (dt[z * t->x * t->y + i * t->y + j]);
				fprintf(f, "%d", px & 0xff);
				if (j < t->y - 1)fprintf(f, " ");
			}
			if (z < t->z - 1)fprintf(f, " 0 ");
		}
		if (i < t->x - 1)fprintf(f, "\n");
	}

	free_mem(dt);
	fclose(f);
	return 0;
}





