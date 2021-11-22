#include <png/png.h>
#include <math.h>
#include "tensor/tensor.h"
#include "gpu/Gpu.h"
#include "gpu/Kernel.h"

#define get_global_id(x) 0

REAL mediaram(Tensor inp);

KV im(Vector image, int ix, int iy, int k0) {
	int k = get_global_id(0) + k0;
	int x, y;
	KTensorRemap2D(k, x, y, iy)
	image[KTensorMap(x, y, 0, ix, iy)] = 255.0 * ((REAL) !(x & y));
	image[KTensorMap(x, y, 1, ix, iy)] = (REAL) (x ^ y);
	image[KTensorMap(x, y, 2, ix, iy)] = (REAL) (x | y);
}


int rgbimage() {
	Gpu gpu = Gpu_new();
	Ecx ecx = Ecx_new(10);
	ecx->addstack(ecx, "clCreateCommandQueueWithProperties");
	cl_command_queue queue = clCreateCommandQueueWithProperties(gpu->context, gpu->device, NULL, &ecx->error);
	ecx->popstack(ecx);
	Tensor timage = Tensor_new(255, 255, 3, 1, ecx, 0, gpu->context, queue);

//	for (int i = 0; i < timage->x*timage->y; ++i) {
//		im(timage->data,timage->x,timage->y,i);
//	}
	char *prog = UTILS_MACRO_KERNEL
				 "KV im(Vector image, int ix, int iy, int k0) {\n"
				 "\tint k = get_global_id(0) + k0;\n"
				 "\tint x,y;\n"
				 "\tKTensorRemap2D(k,x,y,iy)\n"
				 "\timage[KTensorMap(x,y,0,ix,iy)] =  255.0 * ((REAL)!(x & y));\n"
				 "\timage[KTensorMap(x,y,1,ix,iy)] = (REAL)(x^y);\n"
				 "\timage[KTensorMap(x,y,2,ix,iy)] = (REAL)(x|y);\n"
				 "}";
	gpu->compileProgram(gpu, prog);
	Kernel k = Kernel_new(gpu->program, "im", 4, KP, KI, KI, KI, KI);
	ecx->addstack(ecx, "runRecursive");
	k->runRecursive(k, queue, timage->x * timage->y, gpu->maxworks, &timage->data, &timage->x, &timage->y);
	timage->erro->error = k->error;
	ecx->popstack(ecx);
	clFinish(queue);

	unsigned char *image = calloc(timage->x * timage->y, 3);
	char *tjson = timage->json(timage, 2);
	FILE *f = fopen("../a.json", "w");
	fprintf(f, "%s\n", tjson);
	free(tjson);
	fclose(f);

	timage->imagegray(timage, image, timage->y, timage->x, timage->y, timage->x, 0, 0, 0, 0);
	timage->imagegray(timage, image + (timage->x * timage->y), timage->y, timage->x, timage->y, timage->x, 0, 0, 1, 0);
	timage->imagegray(timage, image + (2 * timage->x * timage->y), timage->y, timage->x, timage->y, timage->x, 0, 0, 2, 0);
	pngRGB("../rgb.png", image, image + timage->x * timage->y, image + 2 * timage->x * timage->y, timage->x, timage->y);
	free(image);
	timage->release(&timage);
	gpu->release(&gpu);
	k->release(&k);
	ecx->print(ecx);
	ecx->release(&ecx);
	return 0;
}

int somagpu(Tensor soma,Tensor inp, Kernel k, size_t maxglob);
int somagpu2(Tensor soma,Tensor inp, Kernel k, size_t maxglob);

REAL mediagpu2(Tensor inp, Kernel mean);

#include<sys/time.h>
#include <locale.h>

double us(void);
double t0 = 0;
#define  tic() t0 = us();
#define tac(format,...)t0 = us() - t0; printf(format,##__VA_ARGS__); printf(" %lf us\n",t0);
int main() {
	system("chcp 65001");
	Gpu gpu = Gpu_new();
	Ecx ecx = Ecx_new(10);
	ecx->addstack(ecx, "gpu->Queue_new");
	cl_command_queue queue = gpu->Queue_new(gpu, ecx->perro);
	ecx->popstack(ecx);
	uint32_t x = 280;
	uint32_t y = 280;
	uint32_t z = 8;
	Tensor t_gpu = Tensor_new(x, y, z, 1, ecx, 0, gpu->context, queue);
	Tensor soma = Tensor_new(1, 1, z, 1, ecx, 0, gpu->context, queue);

	char *prog = UTILS_MACRO_KERNEL

				 "KV sum1(Vector soma,unsigned int xy_len, unsigned int batch, int salto, int k0) {\n"
				 "\tint k = get_global_id(0) + k0 ;\n"
				 "\tint z;\n"
				 "\tint xy;\n"
				 "\tKTensorRemap2D(k, z, xy, batch)\n"
				 "xy = xy+salto;"
				 "\tsoma[z*xy_len+xy] = soma[z*xy_len+xy] + soma[z*xy_len +xy+batch] ; \n"
				 "}"
				 ""
				 "kV sum0(Vector exponent, Vector soma,int saidatx, int saidaty, int k0) {\n"
				 "\tint z = get_global_id(0) + k0;\n"
				 "\tint x, y;\n"
				 "\tREAL sum=0;\n"
//				 "printf(\"%d %d\\n\",saidatx,saidaty);"
				 "\tfor (x = 0; x < saidatx; x++)\n"
				 "\t\tfor (y = 0; y < saidaty; y++) {\n"
				 "\t\t\tsum = sum + exponent[KTensorMap(x, y, z, saidatx, saidaty)];\n"
				 "\t\t}\n"
				 "\tsoma[z] = sum;\n"
//				 "printf(\"%d %f\\n\",z,sum);"
				 "}";
	gpu->compileProgram(gpu, prog);
	Kernel k = Kernel_news(gpu->program, "sum1", "Vector soma,unsigned int xy_len, unsigned int batch, int salto, int k0");
	Kernel k2 = Kernel_news(gpu->program, "sum0", "Vector exponent, Vector soma,int saidatx, int saidaty, int k0");

//	t_gpu->randomize(t_gpu, TENSOR_NORMAL, 1, 2);
	REAL fil = 1;
	t_gpu->fillM(t_gpu,0,t_gpu->bytes,&fil, sizeof(REAL));
	t_gpu->print(t_gpu);
	tic()
	somagpu(soma,t_gpu,k,gpu->maxworks);
	tac("Método paralelo")
	soma->print(soma);
	tic()
	somagpu2(soma,t_gpu,k2,gpu->maxworks);
	tac("Método sequencial")
	soma->print(soma);

	t_gpu->release(&t_gpu);
	soma->release(&soma);
	gpu->release(&gpu);
	k->release(&k);
	ecx->print(ecx);
	int e = ecx->error;
	ecx->release(&ecx);
	return e;
}



#include <windows.h>

double us() {
	FILETIME ft;
	LARGE_INTEGER li;
	GetSystemTimePreciseAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	u_int64 ret = li.QuadPart;
	return (double) ret / 10.0;
}

int somagpu(Tensor soma,Tensor inp, Kernel k, size_t maxglob) {
	int len = inp->x*inp->y;
	Tensor aux = Tensor_new(inp->x,inp->y,inp->z,1,inp->erro,inp->flag.flag,inp->context, inp->queue);
	aux->copy(aux,inp);
	int xylen =  inp->x*inp->y;
	int salto = 0;
	while(len>1){
		salto = len%2;
		len = len/2;
		k->runRecursive(k,inp->queue,len*inp->z,maxglob,
						&aux->data,&xylen,&len,&salto
						);
		len = len + salto;
	}
	clFlush(inp->queue);
	for (int i = 0; i < inp->z; ++i) {
		soma->copyM(soma,aux,i*sizeof(REAL),i*xylen*sizeof(REAL),sizeof(REAL));
	}
	clFinish(inp->queue);
	aux->release(&aux);
	return k->error;
}

int somagpu2(Tensor soma, Tensor inp, Kernel k, size_t maxglob) {
//	k->runRecursive(k,inp->queue,inp->z,maxglob,&inp->data,&soma->data,&inp->x,&inp->y);
	clSetKernelArg(k->kernel,0,sizeof(void *),&inp->data);
	clSetKernelArg(k->kernel,1,sizeof(void *),&soma->data);
	clSetKernelArg(k->kernel,2,sizeof(cl_int),&inp->x);
	clSetKernelArg(k->kernel,3,sizeof(cl_int),&inp->y);
	int k0 = 0;
	clSetKernelArg(k->kernel,4,sizeof(cl_int),&k0);
	size_t globals = inp->z;
	soma->erro->error = clEnqueueNDRangeKernel(soma->queue,k->kernel,1,NULL,&globals,&globals,0,NULL,NULL);
//	clFlush(inp->queue);
	clFinish(inp->queue);
	return k->error;
}
