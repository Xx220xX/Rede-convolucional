//
// Created by hslhe on 01/09/2021.
//

#include "gpu/WrapperCL.h"
#include "gpu/Kernel.h"
#include "tensor/Tensor.h"
#include "commonsTest.h"


void testGPUsvm(WrapperCL *cl, QUEUE queue, int m) {
	Tensor a, b, c;
	CNN_ERROR error = {0};
	a = new_Tensor(cl->context, queue, 0, m, 1, 1, 1, &error, NULL);
	b = new_Tensor(cl->context, queue, 0, m, 1, 1, 1, &error, NULL);
	c = new_Tensor(cl->context, queue, 0, m, 1, 1, 1, &error, NULL);
	Kernel soma = new_Kernel(cl->program, &error, soma, 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	TensorFillDouble(queue, a, 1.0);
	TensorFillDouble(queue, b, 2.0);

	printf("%d %zu\n", m, cl->maxworks);
	kernel_run_recursive(error.error, soma, queue, m, cl->maxworks, K_ARG a->data, K_ARG b->data, K_ARG c->data);
	synchronizeKernel(queue);
	double *mem = alloc_mem(c->bytes, 1);
	TensorGetValues(queue,c,mem);
	for (int i = 0; i < m; i++) {
		printf("%lf ", mem[i]);
	}
	free_mem(mem);

	releaseKernel(&soma);
	releaseTensor(&a);
	releaseTensor(&c);
	releaseTensor(&b);

}

void testGPU(WrapperCL *cl, QUEUE queue, int m) {
	Tensor a, b, c;
	CNN_ERROR error = {0};
	a = new_Tensor(cl->context, queue, 0, m, 1, 1, 1, &error, NULL);
	b = new_Tensor(cl->context, queue, 0, m, 1, 1, 1, &error, NULL);
	c = new_Tensor(cl->context, queue, 0, m, 1, 1, 1, &error, NULL);
	Kernel soma = new_Kernel(cl->program, &error, soma, 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	TensorFillDouble(queue, a, 1.0);
	TensorFillDouble(queue, b, 2.0);

	printf("%d %zu\n", m, cl->maxworks);
	kernel_run_recursive(error.error, soma, queue, m, cl->maxworks, K_ARG a->data, K_ARG b->data, K_ARG c->data);

	double *tmp = alloc_mem(m, sizeof(double));

	TensorGetValues(queue, c, tmp);
	for (int i = 0; i < m; i++) {
		printf("%lf ", tmp[i]);
	}
	printf("\n");
	free_mem(tmp);


	releaseKernel(&soma);
	releaseTensor(&a);
	releaseTensor(&c);
	releaseTensor(&b);

}

int main() {
	WrapperCL cl = {0};
	cl.type_device = CL_DEVICE_TYPE_GPU;
	WrapperCl_init(&cl, "__kernel void soma(__global double * A,__global double * B,__global double * C,int kin){"
						"int k0 = get_global_id(0)+kin;"
						"C[k0] = A[k0]+B[k0];"
						"printf(\"%lf \",C[k0]);"
						"}");

	QUEUE queue = clCreateCommandQueueWithProperties(cl.context, cl.device, NULL, NULL);
	testGPUsvm(&cl, queue, 10);
	clReleaseCommandQueue(queue);
	WrapperCL_release(&cl);
	return 0;
}