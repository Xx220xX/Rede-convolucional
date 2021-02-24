#ifndef LIBRARY_LIBRARY_H
#define LIBRARY_LIBRARY_H

//#define LOG_CNN_SALVE_LAYERS
//#define LOG_CNN_TENSOR_MEMORY
//#define LOG_CNN_KERNELCALL
#include "src/cnn.h"

#define REQUEST_INPUT 0
#define REQUEST_GRAD_INPUT 1
#define REQUEST_OUTPUT 2
#define REQUEST_WEIGTH 3

typedef struct {
    void *p;
} Pointer;

void createCnnWrapper(Pointer *p, char *kernelFile, double hitLearn, double momento, double decaimentoDePeso, double multiplicador, UINT inx, UINT iny, UINT inz);

void releaseCnnWrapper(Pointer *p);

int CnnGetSize(Cnn c, int layer, int request, int *x, int *y, int *z, int *n);

int CnnGetTensorData(Cnn c, int layer, int request, int nfilter, double *dest);

int CnnSaveInFile(Cnn c, char *fileName);
int CnnLoadByFile(Cnn c,char *fileName);
void CnnInfo(Cnn c);

int openFILE(Pointer *p, char *fileName, char *mode);
int closeFile(Pointer *p);


int getCnnError(Cnn c);
void getCnnErrormsg(Cnn c,char *msg);
void initRandom(long long int seed);
#endif //LIBRARY_LIBRARY_H
