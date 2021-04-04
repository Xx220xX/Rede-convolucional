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

#endif //CNN_GPU_UTEISTREINO_H
