#ifndef LIBRARY_LIBRARY_H
#define LIBRARY_LIBRARY_H


#include "cnn.h"
#include "cnn/utils/manageTrain.h"

typedef struct {
	void *p;
} Pointer;

void createManageTrainPy(ManageTrain *self, char *luafile, double tx_aprendizado, double momento, double decaimento);

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



#endif //LIBRARY_LIBRARY_H
