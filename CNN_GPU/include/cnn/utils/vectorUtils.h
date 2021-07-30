//
// Created by Henrique on 28-Jul-21.
//

#ifndef CNN_GPU_VECTORUTILS_H
#define CNN_GPU_VECTORUTILS_H

#include <stdio.h>
#include "cnn.h"

void ppmp2(double *data, int x, int y, char *fileName);

void ppmp3(double *data, int x, int y, int z, const char *fileName);

void salveCnnOutAsPPM(Cnn c, const char *name);


int salveTensorAsPPM3(const char *name, Tensor t, Cnn c);

int salveTensor4DAsPPM3(const char *name, Tensor t, Cnn c, UINT w);

void salveTensorAsPPM(const char *name, Tensor t, Cnn c);

int salveTensor4DAsPPM(const char *name, Tensor t, Cnn c, UINT w);


#endif //CNN_GPU_VECTORUTILS_H
