//
// Created by Henrique on 4/2/2021.
//

#ifndef CNN_GPU_UTEISTREINO_H
#define CNN_GPU_UTEISTREINO_H

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

int readBytes(FILE *f, unsigned char *buff, size_t bytes, size_t *bytesReaded) {
	if (feof(f)) {
		fseek(f, 0, SEEK_SET);
		return 1;
	}
	size_t a = 0;
	if (!bytesReaded) {
		bytesReaded = &a;
	}
	*bytesReaded = fread(buff, 1, bytes, f);
	return 0;
}

int normalizeImage(double *imagem, size_t bytes, WrapperCL *cl, cl_command_queue queue, Kernel divInt, FILE *f,
                   size_t *bytesReadd) {

	if (readBytes(f, (unsigned char *) imagem, bytes, bytesReadd)) {
		return 1;
	}
	cl_mem mInt, mDou;
	int error = 0;
	mInt = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes, NULL, &error);
	mDou = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes * sizeof(double), NULL, &error);
	dividirVetorInt((unsigned char *) imagem, imagem, mInt, mDou, bytes, 255, divInt, queue);
	clReleaseMemObject(mInt);
	clReleaseMemObject(mDou);
	return 0;
}

int loadTargetData(double *target, size_t bytes, WrapperCL *cl, cl_command_queue queue, Kernel int2vector, FILE *f,
                   size_t *bytesReadd) {
	if (readBytes(f, (unsigned char *) target, bytes, bytesReadd) || !*bytesReadd)
		return 1;
	cl_mem mInt, mDou;
	int error = 0;
	mInt = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes, NULL, &error);
	mDou = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes * 10 * sizeof(double), NULL, &error);
	int2doubleVector(cl, (unsigned char *) target, target, mInt, mDou, *bytesReadd, 10, int2vector, queue);
	clReleaseMemObject(mInt);
	clReleaseMemObject(mDou);
	return 0;
}


void salveTernsorAsPPM(const char *name, Tensor t, Cnn c) {
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

	free(dt);
	fclose(f);
}

void salveCnnOutAsPPM(const char *name, Cnn c) {
	FILE *f = fopen(name, "wb");
	int maxH = 0;
	int maxW = 0;
	int w, h;
	double *dt;
	Tensor t;
	unsigned char px;
	for (int cm = 0; cm < c->size; cm++) {
		t = c->camadas[cm]->saida;
		w = t->y;
		h = t->x;
		if (t->x > t->y) {
			w = t->x;
			h = t->y;
		}
//		w = w * t->z;
		w = w * t->z + t->z - 1;
		maxH += h;
		if (maxW < w)maxW = w;
	}
	maxW +=2;
	maxH = maxH + c->size;
	fprintf(f, "P5 ");
	fprintf(f, "%d %d ", maxW , maxH);
	fprintf(f, "255 ");

	char *image = calloc(maxW,maxH);
	int imi=0, imj=0;
	for (int cm = 0; cm < c->size; cm++) {
		t = c->camadas[cm]->saida;
		w = t->y;
		h = t->x;
		if (t->x > t->y) {
			w = t->x;
			h = t->y;
		}
		dt = calloc(t->bytes, 1);
		TensorGetValues(c->queue, t, dt);
		normalizeGPU(c, dt, dt, t->bytes / sizeof(double), 255, 0);
		for (int i = 0; i < h; i++) {
			imj=1;
			//image[imi*maxW+imj++] = 0xff;
			for (int z = 0; z < t->z; ++z) {
				for (int j = 0; j < w; ++j) {
					px = ((unsigned int) (dt[z * t->x * t->y + i * t->y + j])) & 0xff;
					image[imi*maxW+imj++]=px;

				}
				//image[imi*maxW+imj++]=0xff;
				imj++;
			}
			imi++;

		}

		//memset(image+imi*maxW,255,maxW);
		imi++;
		free(dt);
	}
	fwrite(image,maxH,maxW,f);
	free(image);
	fclose(f);
}

#endif //CNN_GPU_UTEISTREINO_H
