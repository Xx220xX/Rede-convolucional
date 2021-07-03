#ifndef LIBRARY_LIBRARY_H
#define LIBRARY_LIBRARY_H

//#define LOG_CNN_SALVE_LAYERS
//#define LOG_CNN_TENSOR_MEMORY
//#define LOG_CNN_KERNELCALL
#include "cnn.h"

#define REQUEST_INPUT 0
#define REQUEST_GRAD_INPUT 1
#define REQUEST_OUTPUT 2
#define REQUEST_WEIGTH 3

typedef struct {
	void *p;
} Pointer;

void createCnnPy(Pointer *p, double hitLearn, double momento, double decaimentoDePeso,
                UINT inx, UINT iny, UINT inz);

void releaseCnnWrapper(Pointer *p);

int CnnSaveInFile(Cnn c, char *fileName);

char *camadaToString(Camada c);

int CnnLoadByFile(Cnn c, char *fileName);

int openFILE(Pointer *p, char *fileName, char *mode);

int closeFile(Pointer *p);


void initRandom(long long int seed);

void Py_getCnnOutPutAsPPM(Cnn c, Pointer *p, size_t *h, size_t *w);

void freeP(void *p);


#endif //LIBRARY_LIBRARY_H
