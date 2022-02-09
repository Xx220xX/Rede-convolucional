//
// Created by Henrique on 03/01/2022.
//

#include <math.h>
#include "cnn/cnn_lua.h"
#include "camadas/all_camadas.h"
#include "windows.h"

double seconds() {
	FILETIME ft;
	LARGE_INTEGER li;
	GetSystemTimePreciseAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	u_int64 ret = li.QuadPart;
	return (double) ret * 1e-7;
}

#include "weigths.h"
void debugT(const char *name, Tensor t);
void debugTlin(const char *name, Tensor t);



int main() {
	Cnn c = Cnn_new();
	c->setInput(c, 5, 5, 3);
	c->ConvolucaoF(c, P2D(1, 1), P3D(3, 3, 2), FRELU, 1,1,1,1, Params(0.001, 0.9), RDP(-1));
	c->Pooling(c, P2D(1, 1), P2D(2, 2), MAXPOOL);
	c->ConvolucaoF(c, P2D(1, 1), P3D(3, 3, 3), FRELU, 0,0,0,0, Params(0.001, 0.9), RDP(-1));
//	c->Pooling(c, P2D(1, 1), P2D(2, 2), MAXPOOL);
	c->FullConnect(c, 10, Params(0.001, 0.9), FATIVACAO(FSOFTMAX, 1e-7), RDP(-1), RDP(-1));
	Tensor entrada = Tensor_new(c->size_in.x,c->size_in.y,c->size_in.z,1,c->ecx,TENSOR_RAM);
	Tensor target = Tensor_new(c->getSizeOut(c).x,c->getSizeOut(c).y,c->getSizeOut(c).z,1,c->ecx,TENSOR_RAM);


	CamadaConvF l0 = Conv2D(c->cm[0]);
	CamadaPool l1 = CST_POOL(c->cm[1]);
	CamadaConvF l2 = Conv2D(c->cm[2]);
//	CamadaPool l3 = CST_POOL(c->cm[3]);
	CamadaFullConnect l4 = Dense(c->cm[3]);

	// inicializando pesos
	setl0(l0->w, l0->b);
	setl3(l2->w, l2->b);
	setl7(l4->w, l4->b);
	// inicialização das entradas
	ioKeras(entrada,target);

	// fazer a predição
	c->predict(c,entrada);

//	c->print(c, "#");
// convolucao
//	debugT("l0_conv2_a",l0->super.a);
//	debugT("l0_conv2_W",l0->w);
//	debugT("l0_conv2_B",l0->b);
//	debugT("l0_conv2_z",l0->z);
//	debugT("l0_conv2_s",l0->super.s);

// pooling
//	debugT("l1_pool_s",l1->super.s);
// convolucao
//	debugT("l2_conv2_W",l2->w);
//	debugT("l2_conv2_B",l2->b);
//	debugT("l2_conv2_z",l2->z);
	debugT("l2_conv2_s",l2->super.s);
// pooling
//	debugTlin("l3_pool_s",l3->super.s);
// dense
	debugT("l4_dense_a",l4->super.a);
	debugT("l4_dense_w",l4->w);
	debugT("l4_dense_b",l4->b);
	debugTlin("l4_dense_z",l4->z);
//	debugT("l4_dense_s",l4->super.s);

// calcula os gradientes
	c->learnBatch(c,target,1);
	debugTlin("db4",l4->db);
	debugT("dw4",l4->dw);
	debugTlin("dw2",l2->dw);
	debugTlin("db2",l2->db);
	debugTlin("dw0",l0->dw);
	debugTlin("db0",l0->db);
	Release(entrada);
	Release(target);
	return c->release(&c);
}
void debugT(const char *name, Tensor t){
	char buf[250]="";
	snprintf(buf,250,"../debug/txt/C_%s.txt",name);
	FILE *f = fopen(buf, "w");
	fprintf(f, "%s(%zu,%zu,%zu,%zu)",name, t->x, t->y, t->z,t->w);
	t->fprint(t, f);
	fclose(f);
}

void debugTlin(const char *name, Tensor t) {
	char buf[250]="";
	REAL v[t->length];
	t->getvalues(t,v);
	snprintf(buf,250,"../debug/py/C_%s.py",name);
	FILE *f = fopen(buf, "w");
	fprintf(f,"%s = [%.9g",name,v[0]);
	for(int i=1;i<t->length;i++){
		fprintf(f,", %.9g",v[i]);
	}
	fprintf(f, "]\n");
	fprintf(f,"# (%zu, %zu, %zu, %zu)\n",t->x,t->y,t->z,t->w);
	fclose(f);
}
