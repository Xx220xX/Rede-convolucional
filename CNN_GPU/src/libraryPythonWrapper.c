#include "libraryPythonWrapper.h"
#include "cnn/utils/defaultkernel.h"

void PY_createCnn(Cnn c, double hitLearn, double momento, double decaimentoDePeso,
				  UINT inx, UINT iny, UINT inz) {
	Params pr = {hitLearn, momento, decaimentoDePeso};
	WrapperCL *cl = (WrapperCL *) alloc_mem(sizeof(WrapperCL), 1);
	cl->type_device = CL_DEVICE_TYPE_GPU;
	WrapperCl_init(cl, default_kernel);
	Cnn tmp = createCnn(cl, pr, inx, iny, inz);
	memcpy(c, tmp, sizeof(Cnn_t));
	c->releaseCL = 1;
	c->release_self = 0;
	free_mem(tmp);

}

void PY_releaseCnn(Cnn c) {
	releaseCnn(&c);
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


void initRandom(long long int seed) {
	LCG_setSeed(seed);
}


void Py_getCnnOutPutAsPPM(Cnn c, String *p, size_t *h, size_t *w) {
	releaseStr(p);
	p->d = salveCnnOutAsPPMGPU(c, h, w);
	p->size =*w**h;
	p->release=1;
}


void createManageTrainPy(ManageTrain *self, char *luafile, double tx_aprendizado, double momento, double decaimento) {
	*self = createManageTrain(luafile, tx_aprendizado, momento, decaimento);
}
