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

void ppmp3(double *data, int x, int y, int z, char *fileName) {
	FILE *f = fopen(fileName, "w");
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", y, x);
	fprintf(f, "255\n");
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; ++j) {
			for (int k = 0; k < z; ++k) {
				fprintf(f, "%d", (int) (data[k * y * x + i * y + j] * 255));
				if (k < z - 1)fprintf(f, " ");
			}
			if (j < y - 1)fprintf(f, " ");
		}
		if (i < x - 1)
			fprintf(f, "\n");
	}
	fclose(f);
}


int normalizeImage(Cnn cnn, double *imagem, size_t bytes, FILE *f, size_t *bytesReadd) {
	if(cnn->error.error)return cnn->error.error;
	if (readBytes(f, (unsigned char *) imagem, bytes, bytesReadd)) {
		cnn->error.error = -200;
		snprintf(cnn->error.msg, EXCEPTION_MAX_MSG_SIZE, "normalizeImage, load %d of %d\n", *bytesReadd, bytes);
		return cnn->error.error;
	}

	Tensor mInt, mDou;

	mInt = new_Tensor(cnn->cl->context, cnn->queue, TENSOR_UHST | TENSOR_CHAR, bytes, 1, 1, 1, &cnn->error, imagem);
	mDou = newTensor(cnn->cl->context, cnn->queue, bytes, 1, 1, TENSOR_NCPY, &cnn->error);


	if (cnn->error.error) {
		releaseTensor(&mInt);
		releaseTensor(&mDou);
		fprintf(stderr, "Erro on normalizeImage\n");
		return cnn->error.error;
	}
	double value = 255;
	kernel_run_recursive(cnn->error.error, cnn->kerneldivInt, cnn->queue, bytes, cnn->cl->maxworks,
	                     K_ARG mInt->data,
			K_ARG mDou->data,
			K_ARG value
	);
	if (cnn->error.error) {
		getClErrorWithContext(cnn->error.error, cnn->error.msg, EXCEPTION_MAX_MSG_SIZE,
		                      "normalizeImage/cnn->kerneldivInt:");
		releaseTensor(&mInt);
		releaseTensor(&mDou);
		return cnn->error.error;
	}
	releaseTensor(&mInt);
	cnn->error.error = TensorGetValuesMem(cnn->queue, mDou, imagem, bytes);
	if (cnn->error.error) {
		getClErrorWithContext(cnn->error.error, cnn->error.msg, EXCEPTION_MAX_MSG_SIZE,
		                      "normalizeImage/TensorgetValueMem:");

	}
	releaseTensor(&mDou);

	return cnn->error.error;
}

int loadTargetData(Cnn cnn, double *target, unsigned char *labelchar, int numeroDeClasses, size_t bytes, FILE *f,
                   size_t *bytesReadd) {
	if(cnn->error.error)return cnn->error.error;
	if (labelchar == NULL)labelchar = (unsigned char *) target;
	if (readBytes(f, labelchar, bytes, bytesReadd) || !*bytesReadd){
		cnn->error.error = -200;
		snprintf(cnn->error.msg, EXCEPTION_MAX_MSG_SIZE, "normalizeImage, load %d of %d\n", *bytesReadd, bytes);
		return cnn->error.error;
	}
	Tensor mInt, mDou;
	mInt = new_Tensor(cnn->cl->context, cnn->queue, TENSOR_UHST | TENSOR_CHAR, bytes, 1, 1, 1, &cnn->error, labelchar);
	mDou = newTensor(cnn->cl->context, cnn->queue, bytes, numeroDeClasses, 1, TENSOR_NCPY, &cnn->error);
	if (cnn->error.error) {
		fprintf(stderr, "loadTargetData: ");
		getClErrorWithContext(cnn->error.error, cnn->error.msg, EXCEPTION_MAX_MSG_SIZE,
		                      "loadTargetData/newTensor:");
		releaseTensor(&mInt);
		releaseTensor(&mDou);
		return cnn->error.error;
	}
	kernel_run_recursive(cnn->error.error,cnn->kernelInt2Vector,cnn->queue,
	                     bytes,cnn->cl->maxworks,
	                     K_ARG mInt->data,
			K_ARG mDou->data,
			K_ARG numeroDeClasses
	);

	if (cnn->error.error) {
		getClErrorWithContext(cnn->error.error, cnn->error.msg, EXCEPTION_MAX_MSG_SIZE,
		                      "loadTargetData/cnn->kernelInt2Vector:");
		releaseTensor(&mInt);
		releaseTensor(&mDou);
		return cnn->error.error;
	}
	releaseTensor(&mInt);
	cnn->error.error = TensorGetValuesMem(cnn->queue, mDou, target, bytes);
	if (cnn->error.error) {
		getClErrorWithContext(cnn->error.error, cnn->error.msg, EXCEPTION_MAX_MSG_SIZE,
		                      "loadTargetData/TensorgetValueMem:");
	}
	releaseTensor(&mDou);
	return cnn->error.error;
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
	double *dt = calloc(t->bytes, 1);
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

int salveTensor4DAsPPM(const char *name, Tensor t, Cnn c, UINT w) {
	if (w >= t->w) return -2;

	double *dt = calloc(t->bytes, 1);
	c->error.error = TensorGetValues(c->queue, t, dt);
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





