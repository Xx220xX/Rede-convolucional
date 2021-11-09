#ifndef LIBRARY_LIBRARY_H
#define LIBRARY_LIBRARY_H


#include "cnn.h"
#include "cnn/utils/manageTrain.h"

typedef struct {
	void *p;
} Pointer;

void createManageTrainPy(ManageTrain *self, char *luafile, REAL tx_aprendizado, REAL momento, REAL decaimento);

void createManageTrainPyStr(ManageTrain *self, char *lua_data, REAL tx_aprendizado, REAL momento, REAL decaimento);

void PY_createCnn(Cnn c, REAL hitLearn, REAL momento, REAL decaimentoDePeso,
				  UINT inx, UINT iny, UINT inz);

void PY_releaseCnn(Cnn c);

int CnnSaveInFile(Cnn c, char *fileName);


int CnnLoadByFile(Cnn c, char *fileName);

void initRandom(long long int seed);

void Py_getCnnOutPutAsPPM(Cnn c, String *p, size_t *h, size_t *w);


#endif //LIBRARY_LIBRARY_H
