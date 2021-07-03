//
// Created by Henrique on 28-May-21.
//


void initPy();

#include "../src/cnn.h"
#include <python/Python.h>

PyObject *TensorToPython(Tensor t, cl_command_queue queue);

void plotTensor(Tensor t, cl_command_queue queue, char *title);

#define X_ 2
#define Y_ 1
#define Z_ 1
#define SIM X_*Y_*Z_
#include "camada/batchnorm.h"

int main() {
	initPy();

	Params p = {0.1, 0.0, 0.0, 1};
	Cnn c = createCnnWithWrapper(KERNEL_FILE, p, X_,Y_,Z_, CL_DEVICE_TYPE_GPU);
	double entrada[SIM]={1.0,1.0};
	double saida[1]={0};
	LCG_setSeed(time(0));
	CnnAddFullConnectLayer(c,5,FSIGMOID);
//	for (int i = 0; i < SIM; i++) {
//		entrada[i] = LCG_randD() * 200 - 100;
//		saida[i] = LCG_randD() * 2- 1;
//	}

//	for (int i = 0; i < SIM; printf("%g ",entrada[i++])) ;printf("\n");
	run(c, entrada,saida);
	PyRun_SimpleString("plt.show()");
	releaseCnn(&c);
	Py_Finalize();
}

void initPy() {
	Py_Initialize();
	PyRun_SimpleString(
			"import numpy as np\n"
			"import ctypes as c\n"
			"from threading import Thread\n"
			"import matplotlib.pyplot  as plt\n"
			"def plot1d(data,title):\n"
			"   plt.figure()\n"
			"   plt.stem(data)\n"
			"   plt.title(title)\n"
			"   plt.text(0,0,'%s %d'%(title,len(data)))\n"
			"   \n");
}

PyObject *TensorToPython(Tensor t, cl_command_queue queue) {
	double *v = calloc(t->bytes, 1);

	TensorGetValues(queue, t, v);
	size_t size = t->x * t->y * t->z;
	PyObject *list = PyList_New(size);
	for (int i = 0; i < size; i++) {
		PyList_SetItem(list, i, PyFloat_FromDouble(v[i]));
	}
	free(v);
	return list;
}

void plotTensor(Tensor t, cl_command_queue queue, char *title) {
	PyObject *list = TensorToPython(t, queue);
	PyObject *moduleMainString = PyUnicode_FromString("__main__");
	PyObject *moduleMain = PyImport_Import(moduleMainString);
	PyObject *func = PyObject_GetAttrString(moduleMain, "plot1d");
	PyObject *args = PyTuple_Pack(2, list, PyUnicode_FromString(title));
	PyObject *result = PyObject_CallObject(func, args);

}

