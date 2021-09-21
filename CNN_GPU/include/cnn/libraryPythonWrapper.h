#ifndef LIBRARY_LIBRARY_H
#define LIBRARY_LIBRARY_H


#include "cnn.h"
#include "cnn/utils/manageTrain.h"

typedef struct {
	void *p;
} Pointer;

void createManageTrainPy(ManageTrain *self, char *luafile, double tx_aprendizado, double momento, double decaimento);

void createManageTrainPyStr(ManageTrain *self, char *lua_data, double tx_aprendizado, double momento, double decaimento);

void PY_createCnn(Cnn c, double hitLearn, double momento, double decaimentoDePeso,
				  UINT inx, UINT iny, UINT inz);

void PY_releaseCnn(Cnn c);

int CnnSaveInFile(Cnn c, char *fileName);


int CnnLoadByFile(Cnn c, char *fileName);

void initRandom(long long int seed);

void Py_getCnnOutPutAsPPM(Cnn c, String *p, size_t *h, size_t *w);


#endif //LIBRARY_LIBRARY_H
