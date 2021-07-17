#define __kernel
#define __global
#define get_global_id(x) global_index_
int global_index_ = 0;

#define RUN_KERNEL(kernel, iteractions, v0, ...) \
    for(global_index_ = 0;global_index_ <iteractions;global_index_++) \
                             kernel(v0, ## __VA_ARGS__)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "camadas/utils.h"
#include "camadas/pool.h"
#include "camadas/poolav.h"
#include "camadas/conv.h"
#include "camadas/fullconnect.h"

int testePool() {
	double gradIn[8];
	double entrada[8] = {0, 2, 3, 5, 4, 8, 9, 9};
	double saida[3] = {2, 5, 9};
	int py = 3;
	int fy = 3;
	int iny = 8;
	int sy = 3;
	RUN_KERNEL(poolCalcGrads, 8, entrada, gradIn, saida, saida, 1, fy, 1, py, 1, iny, 1, sy, 0);
	for (int i = 0; i < 8; i++) {
		printf("%lf ", gradIn[i]);
	}
	return 0;
}

typedef struct {
	int x, y, z, w;
	double *data;
} Tensor;

Tensor newTensor(int x, int y,int  z,int w) {
	Tensor t = {x, y, z, w, calloc(x * y * z * w, sizeof(double))};
	return t;
}

void releaseTensor(Tensor *t) {
	if (t->data)free(t->data);
	*t  =(Tensor){0};
}
#define TensorALLDIM(t)t.x*t.y*t.z*t.w
void randTensor(Tensor *t){
	for(int i = TensorALLDIM((*t))-1;i>=0;i--){
		t->data[i]=rand()*2.0/RAND_MAX-1.0;
	}
}
void printPyTensor(FILE *f,Tensor t,char *name){
	fprintf(f,"%s = [",name);
	fprintf(f,"%lf",t.data[0]);
	for(int i=1;i<TensorALLDIM(t);i++){
		fprintf(f,", %lf",t.data[i]);
	}
	fprintf(f,"];\n");
	fprintf(f,"%s = np.array(%s).reshape(%d,%d,%d,%d)\n",name,name,t.x,t.y,t.z,t.w);

}
int main() {
	srand(10);
	rand();
	int px=1,py=1;
	Tensor filtro =  newTensor(2,2,2,2);
	Tensor entrada = newTensor(2,2,2,1);
	Tensor saida = newTensor((entrada.x-filtro.x)/px+1,
							 (entrada.y-filtro.y)/py+1,
							 filtro.w,1);
	Tensor gradfiltro =  newTensor(2,2,2,2);
	Tensor gradentrada = newTensor(2,2,2,1);
	Tensor target = newTensor(saida.x,saida.y,saida.z,saida.w);
	Tensor ds = newTensor(saida.x,saida.y,saida.z,saida.w);
	randTensor(&filtro);
	randTensor(&entrada);
	randTensor(&target);

	RUN_KERNEL(convSum, TensorALLDIM(saida),filtro.data,entrada.data,
			   saida.data,px,saida.x,saida.y,entrada.x,entrada.y,filtro.x,entrada.z,0);
	// print
	FILE *pyfile =fopen("../conv.py","w");
	fprintf(pyfile,"import numpy as np\n");
	printPyTensor(pyfile,entrada,"a");
	printPyTensor(pyfile,filtro,"f");
	printPyTensor(pyfile,target,"t");
	printPyTensor(pyfile,saida,"s");
	// aprendizado
	RUN_KERNEL(sub, TensorALLDIM(ds),ds.data,saida.data,target.data,0);
	printPyTensor(pyfile,ds,"ds");

	fclose(pyfile);
	// release
	{
		releaseTensor(&entrada);
		releaseTensor(&filtro);
		releaseTensor(&saida);
		releaseTensor(&target);
		releaseTensor(&gradfiltro);
		releaseTensor(&gradentrada);
		releaseTensor(&ds);
	}
}