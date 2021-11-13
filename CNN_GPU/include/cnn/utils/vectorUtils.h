//
// Created by Henrique on 28-Jul-21.
//

#ifndef CNN_GPU_VECTORUTILS_H
#define CNN_GPU_VECTORUTILS_H

#include <stdio.h>
#include "cnn.h"

void ppmp2(REAL *data, int x, int y, char *fileName);

void ppmp3(REAL *data, int x, int y, int z, const char *fileName);

void salveCnnOutAsPPM(Cnn c, const char *name);

void salveCnnOutAsPPMR(Cnn c, const char *name, size_t width, size_t height);

int salveTensorAsPPM3(const char *name, Tensor t, Cnn c);

int salveTensor4DAsPPM3(const char *name, Tensor t, Cnn c, UINT w);

void salveTensorAsPPM(const char *name, Tensor t, Cnn c);

int salveTensor4DAsPPM(const char *name, Tensor t, Cnn c, UINT w);

int dividirVetor(REAL *v, Tensor m, size_t len, REAL value, Kernel funcNorm, size_t max_works,
				 QUEUE queue);

#endif //CNN_GPU_VECTORUTILS_H
