//
// Created by Henrique on 28-Jul-21.
//

#ifndef CNN_GPU_VECTORUTILS_H
#define CNN_GPU_VECTORUTILS_H

#include <stdio.h>
#include "cnn.h"

void ppmp2(double *data, int x, int y, char *fileName);

void salveCnnOutAsPPM(Cnn c, const char *name);



void salveTensorAsPPM(const char *name, Tensor t, Cnn c);

int salveTensor4DAsPPM(const char *name, Tensor t, Cnn c, UINT w);

void ppmp3(double *data, int x, int y, int z, char *fileName);

int readBytes(FILE *f, unsigned char *buff, size_t bytes, size_t *bytesReaded);

int normalizeImage(Cnn cnn, double *imagem, size_t bytes, FILE *f, size_t *bytesReadd);

int loadTargetData(Cnn cnn, double *target, unsigned char *labelchar, int numeroDeClasses,
                   size_t bytes, FILE *f, size_t *bytesReadd);

/**
 * remove se existir um diretorio
 * e recria novamente, vazio
 * @param dir
 */
void createDir(char *dir);
#endif //CNN_GPU_VECTORUTILS_H
