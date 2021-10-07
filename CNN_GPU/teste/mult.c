#include <utils/time_utils.h>
#include "gpu/WrapperCL.h"
#include "gpu/Kernel.h"
#include "tensor/Tensor.h"
#include "commonsTest.h"
#include "stdarg.h"

double multCpu(Tensor a, Tensor b, Tensor c);

double multGpu(QUEUE q, Kernel k, size_t globals, Tensor a, Tensor b, Tensor c);


static double t0 = 0;

void tic() {
	t0 = getms();
}

double tac(const char *format, ...) {
	double v = getms() - t0;
//	va_list arg;
//	va_start(arg, format);
//	vprintf(format, arg);
//	printf(" %.3lf ms", v);
//	printf("\n");
//	va_end(arg);
	return v;
}

int calc(int g, double *tcpu, double *tgpu) {
	char *sum =
			"__kernel void mult(__global double* a,__global double* b,__global double* c,int na,int nc,int k0){"
			"	int w = get_global_id(0) + k0;"
			"   int j = w%nc;"
			"   int i = w/nc;"
			"   double sum = 0;"
			"	for(int k=0;k<na;k++) "
			"		sum+= a[i*na+k]*b[k*nc+j]; "
			"	c[w] = sum;"
//			"c[w] = a[w]+b[w];"
			"}";


	int ma = g;
	int na = g;
	int nb = g;
	WrapperCL cl = {0};
	CNN_ERROR e = {0};
	QUEUE queue = NULL;
	Kernel kmul;

	cl.type_device = CL_DEVICE_TYPE_GPU;
	WrapperCl_init(&cl, sum);
	queue = clCreateCommandQueueWithProperties(cl.context, cl.device, NULL, NULL);
	kmul = new_Kernel(cl.program, &e, mult, 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT);
	LCG_setSeed(124);
	Tensor a;
	Tensor b;
	Tensor c;

	//	CPU
	a = new_Tensor(cl.context, queue, TENSOR_RAM, ma, na, 1, 1, &e, NULL);
	b = new_Tensor(cl.context, queue, TENSOR_RAM, na, nb, 1, 1, &e, NULL);
	c = new_Tensor(cl.context, queue, TENSOR_RAM, ma, nb, 1, 1, &e, NULL);
	tic();
	if (e.error) {
		printf("%s\n", e.msg);
		exit(e.error);
	}
	multCpu(a, b, c);
	*tcpu = tac("CPU mult");
	releaseTensor(&a);
	releaseTensor(&b);
	releaseTensor(&c);

	//	GPU
	a = new_Tensor(cl.context, queue, 0, ma, na, 1, 1, &e, NULL);
	b = new_Tensor(cl.context, queue, 0, na, nb, 1, 1, &e, NULL);
	c = new_Tensor(cl.context, queue, 0, ma, nb, 1, 1, &e, NULL);
	if (e.error) {
		printf("%s\n", e.msg);
		exit(e.error);
	}
	tic();
	multGpu(queue, kmul, cl.maxworks, a, b, c);
	*tgpu = tac("GPU mult");
	releaseTensor(&a);
	releaseTensor(&b);
	releaseTensor(&c);


	releaseKernel(&kmul);
	WrapperCL_release(&cl);
	clReleaseCommandQueue(queue);
	return 0;
}

int main() {
	FILE *f = fopen("../data.m", "w");
	double cpu = 0, gpu = 0;
	fprintf(f, "clc;clear all;close all;");
	fprintf(f, "tm = [");
	for (int i = 1; i <= 800; i+=1) {
		calc(i, &cpu, &gpu);
		fprintf(f, "%.3lf,%.3lf\n", cpu, gpu);
		printf("%d\n", i);
	}
	fprintf(f, "];\n");
	fprintf(f, "plot(tm)\nlegend({'cpu','gpu'})");

	fclose(f);
	return 0;
}

double multCpu(Tensor a, Tensor b, Tensor c) {
	double sum;
	for (int i = 0; i < a->x; ++i) {
		for (int j = 0; j < b->y; ++j) {
			sum = 0;
//			for (int k = 0; k < a->y; ++k) {
//				sum += a->hostd[i * a->y + k] * b->hostd[k * b->y + j];
//			}
//			c->hostd[i * c->y + j] = sum;
			c->hostd[i * c->y + j] = a->hostd[i * c->y + j] + b->hostd[i * c->y + j];
		}
	}
	return 0;
}

double multGpu(QUEUE queue, Kernel k, size_t globals, Tensor a, Tensor b, Tensor c) {
	int erro = 0;
	kernel_run_recursive(erro, k, queue, c->x * c->y, globals, K_ARG a->data, K_ARG b->data, K_ARG c->data, K_ARG a->y, K_ARG c->y);
	synchronizeKernel(queue);
	if (erro)exit(erro);
	return 0;
}
