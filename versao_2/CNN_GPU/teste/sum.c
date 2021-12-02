#include <utils/time_utils.h>
#include "gpu/WrapperCL.h"
#include "gpu/Kernel.h"
#include "tensor/Tensor.h"
#include "commonsTest.h"


int main() {
	char *sum =
			"__kernel void sum(__global double* v,__global double* v,int n,int k0){"
			"	int i = get_global_id(0) + k0;"
			"	v[i] += v[n+i]; "
			"}";
	int n = 10e6;
	WrapperCL cl = {0};
	cl.type_device = CL_DEVICE_TYPE_GPU;
	WrapperCl_init(&cl, sum);
	QUEUE queue = clCreateCommandQueueWithProperties(cl.context, cl.device, NULL, NULL);
	CNN_ERROR e = {0};
	Kernel ksum = new_Kernel(cl.program, &e, sum,3, K_VOID_P, K_INT, K_INT);
	double *data = (double *) calloc(n, sizeof(double));
	LCG_setSeed(124);
	for (int i = 0; i < n; ++i) {
		data[i] = LCG_randn() + 5;
	}

	Tensor cpu = new_Tensor(cl.context, queue, TENSOR_CPY | TENSOR_RAM, n, 1, 1, 1, &e, data);

	double s1 = 0, s2 = 0;
	double t0 = getms();
	s1 = sumcpu(cpu);
	double t1 = getms() - t0;
	printf("end cpu\n");
	releaseTensor(&cpu);
	Tensor gpu = new_Tensor(cl.context, queue, TENSOR_CPY, n, 1, 1, 1, &e, data);
	free(data);
	t0 = getms();
	s2 = sumgpu(gpu,queue,ksum,cl.maxworks);
	double t2 = getms() - t0;
	printf("Tempo na CPU %lf ms result %lf\nTempo na GPU %lf ms result %lf\n", t1, s1, t2, s2);

	releaseKernel(&ksum);
	WrapperCL_release(&cl);
	releaseTensor(&cpu);
	releaseTensor(&gpu);
	return 0;
}


double sumcpu(Tensor v) {
	double soma = 0;
	int n = v->x * v->y * v->z * v->w;
	for (int i = 0; i < n; ++i) {
		soma += v->hostd[i];
	}
	return soma/(v->x * v->y * v->z * v->w);
}

double sumgpu(Tensor v, QUEUE q, Kernel k,size_t globals) {
	double soma = 0;
	int n = v->x * v->y * v->z * v->w;
	int erro = 0;
	int arg1,arg2;
	while (n > 1 && !erro) {
		if (n % 2 == 1) {
			arg1 = n - 1;
			arg2 = 0;
			kernel_run(erro, k, q, 1, 1, K_ARG v->data, K_ARG arg1, K_ARG arg2);
		}
		n = n / 2;

		kernel_run_recursive(erro, k, q, n, globals, K_ARG v->data, K_ARG n);
	}
//	TensorGetValuesMem(q,v,&soma,sizeof (double ));
	synchronizeKernel(q);
	return soma/(v->x * v->y * v->z * v->w);
}
