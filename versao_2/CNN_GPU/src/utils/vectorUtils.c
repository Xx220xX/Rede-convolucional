//
// Created by Henrique on 28-Jul-21.
//

#include "utils/vectorUtils.h"
#include <windows.h>

void ppmp2(REAL *data, int x, int y, char *fileName) {
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

void ppmp3(REAL *data, int x, int y, int z, const char *fileName) {
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

void salveCnnOutAsPPMR(Cnn c, const char *name, size_t width, size_t height) {
	unsigned char *im = alloc_mem(width, height);


	int bytesm = 0;
	REAL *v = NULL;
	Tensor saida = NULL;
	REAL mx, mn;
	int padh = 1;
	int padw = 1;
	int maxz = 0;
	for (int cm = -1; cm < c->size; ++cm) {
		if (cm == -1)saida = c->camadas[0]->entrada;
		else saida = c->camadas[cm]->saida;
		if (maxz < saida->z)maxz = saida->z;
		if(bytesm<saida->bytes)bytesm = saida->bytes;
	}
	v = alloc_mem(bytesm,1);
	int h = height / (c->size + 1) - padh * (c->size + 1);
	int w = width / maxz - padw * maxz;
	int j0, i0;
	double px, py;
	int x, y;
	int vx, vy;
	REAL vr;
	for (int cm = 0; cm < 1; ++cm) {
		if (cm == -1)saida = c->camadas[0]->entrada;
		else saida = c->camadas[cm]->saida;

		TensorGetValues(c->queue, saida, v);
		mx = v[0];
		mn = v[0];
		maxz = 0;
		for (int i = saida->x * saida->y * saida->z - 1; i > 0; i--) {
			if (mx < v[i])mx = v[i];
			if (mn > v[i])mn = v[i];
		}
		mx = mx - mn;
		vx = saida->x;
		vy = saida->y;
		if (vx > vy) {
			vx = saida->y;
			vy = saida->x;
		}
		printTensor(c->queue, saida, stdout);
		for (int z = 0; z < saida->z; ++z) {
			px = vx / (double) h;
			py = vy / (double) w;
			j0 = z * (padw + w);
			i0 = (cm + 1) * (padh + h);
			for (int i = 0; i < h; ++i) {
				for (int j = 0; j < w; ++j) {
					x = i * px;
					y = i * py;
					vr = v[z * vx * vy + x * vy + y];
					printf("%f ", vr);
					im[(i + i0) * width + j + j0] = ((int) ((vr - mn) / mx)) & 0xff;
				}
				printf("\n");
			}

		}

	}
	free_mem(v);


	FILE *f = fopen(name, "wb");
	fprintf(f,
			"P5 ");
	fprintf(f,
			"%zu %zu ", width, height);
	fprintf(f,
			"255 ");
	fwrite(im, height, width, f
	);
	free_mem(im);
	fclose(f);
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
	REAL *dt = alloc_mem(t->bytes, 1);
	FILE *f = fopen(name, "w");
	TensorGetValues(c->queue, t, dt);
	normalizeGPU(c, dt, dt, t->bytes / sizeof(REAL), 255, 0);
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
	REAL *dt = alloc_mem(t->bytes, 1);
	TensorGetValuesOffSet(c->queue, t, dt, t->bytes * w);
	normalizeGPU(c, dt, dt, t->bytes / sizeof(REAL), 255, 0);
	ppmp3(dt, t->x, t->y, t->z, name);
	free_mem(dt);
	return 0;
}

int salveTensor4DAsPPM(const char *name, Tensor t, Cnn c, UINT w) {
	if (w >= t->w) return -2;
	REAL *dt = alloc_mem(t->bytes, 1);
	c->error.error = TensorGetValuesOffSet(c->queue, t, dt, t->bytes * w);
	if (c->error.error) {
		free_mem(dt);
		return c->error.error;
	}
	normalizeGPU(c, dt, dt, t->bytes / sizeof(REAL), 255, 0);
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


int dividirVetor(REAL *v, Tensor m, size_t len, REAL value, Kernel funcNorm, size_t max_works,
				 QUEUE queue) {
	int error = TensorPutValuesMem(queue, m, v, len * sizeof(REAL));
	if (error)return error;
	kernel_run_recursive(error, funcNorm, queue, len, max_works, K_ARG m, K_ARG value);
	if (error)return error;
	error = TensorGetValuesMem(queue, m, v, len * sizeof(REAL));
	return error;
}



