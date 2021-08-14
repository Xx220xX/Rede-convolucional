#include "libraryPythonWrapper.h"
#include "cnn/utils/defaultkernel.h"




void createCnnPy(Pointer *p, double hitLearn, double momento, double decaimentoDePeso,
                       UINT inx, UINT iny, UINT inz) {
	Params pr = {hitLearn, momento, decaimentoDePeso};
	WrapperCL *cl = (WrapperCL *) alloc_mem(sizeof(WrapperCL), 1);
	cl->type_device = CL_DEVICE_TYPE_GPU;
	WrapperCL_init(cl,default_kernel);
	Cnn c = createCnn(cl, pr, inx, iny, inz);
	c->releaseCL = 1;
	p->p = c;
}

void releaseCnnWrapper(Pointer *p) {
	releaseCnn((Cnn *) &p->p);
}


int CnnSaveInFile(Cnn c, char *fileName) {
	FILE *f = fopen(fileName, "wb");
	if (f == NULL)return -1;
	cnnSave(c, f);
	fclose(f);
	return c->error.error;
}

int CnnLoadByFile(Cnn c, char *fileName) {
	FILE *f = fopen(fileName, "rb");
	if (f == NULL)return -1;
	int err = cnnCarregar(c, f);
	fclose(f);
	return err;
}

int openFILE(Pointer *p, char *fileName, char *mode) {
	if (p == NULL)return -2;
	p->p = fopen(fileName, mode);
	if (p->p == NULL)return -1;
	return 0;
}

int closeFile(Pointer *p) {
	if (p == NULL)return -2;
	if (p->p == NULL)return -1;
	fclose(p->p);
	return 0;
}


void initRandom(long long int seed) {
	LCG_setSeed(seed);
}

void putst(char **pdest, char *src) {
	size_t lsrc = strlen(src), ldest = 0;
	char *dest = *pdest;
	if (dest != NULL)ldest = strlen(dest);
	dest = (char *) realloc(dest, lsrc + ldest);
	strcat(dest, src);
	*pdest = dest;
}


void free_memP(void *p) {
	if (p != NULL)
		free_mem(p);
}

void Py_getCnnOutPutAsPPM(Cnn c, Pointer *p, size_t *h, size_t *w) {
	p->p = salveCnnOutAsPPMGPU(c,h,w);
}

char *camadaToString(Camada c) {
	c->toString(c);
	return c->__string__;
}

void createManageTrainPy(ManageTrain *self, char *luafile, double tx_aprendizado, double momento, double decaimento) {
	ManageTrain tmp  = createManageTrain(luafile,tx_aprendizado,momento,decaimento);
	memcpy(self,&tmp,sizeof (ManageTrain));
}
