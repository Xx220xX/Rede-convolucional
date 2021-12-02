//
// Created by hslhe on 26/08/2021.
//
/***
 *  Faz a verificação da memoria compartilhada utilizando GPU e não utilizando
 *  Verifica a velocidade de processamento em uma multiplicação de uma matriz 100x100 por uma 100x100
 *
 *  As matrizes são geradas apenas utilzando a função aleatoria com a semente 99
 *
 */
#include "gpu/WrapperCL.h"
#include "gpu/Kernel.h"
#include "tensor/Tensor.h"
#include "utils/time_utils.h"

Time_proces svm_only_host(QUEUE q, WrapperCL *cl, int m, int n);

Time_proces svm_using_svm(QUEUE q, WrapperCL *cl, int m, int n);

Time_proces svm_GPU_using_svm(QUEUE q, WrapperCL *cl, int m, int n);

Time_proces svm_GPU_using_host(QUEUE q, WrapperCL *cl, int m, int n);

Time_proces svm_GPU(QUEUE q, WrapperCL *cl, int m, int n);

int main() {
	WrapperCL cl = {0};
	cl.type_device = CL_DEVICE_TYPE_GPU;
	WrapperCl_init(&cl, "__kernel void mult(__global double * A,__global double * B,__global double * C,int N,int kin){"
						"int k0 = get_global_id(0)+kin;"
						"int i = k0/N;"
						"int j = k0/N;"
						"double soma=0;"
						"for (int k = 0; k < N; k++) {"
						"soma += A[i*N+k]*B[k*N+j];"
						"}"
						"C[k0] = soma;"
						"}");


	QUEUE queue = clCreateCommandQueueWithProperties(cl.context, cl.device, NULL, NULL);
	Time_proces tempo_only_host;
	Time_proces tempo_svm;
	Time_proces tempo_svmGPU;
	Time_proces tempo_onlyGPU;
	FILE *f = fopen("onlygpuMulttime.csv", "w");
	fprintf(f, "M,Only host,Only host + svm,svm + gpu,only gpu\n");
	fprintf(f, "| ---- | ---- | ---- | ---- | --- |\n");
	int max = 1000;
	for (int M = 1; M < max; M++) {
		tempo_svmGPU = svm_GPU_using_svm(queue, &cl, M, M);
		tempo_onlyGPU = svm_GPU(queue, &cl, M, M);
		printf("%d %lf\n", M, (M + 1.0) / max * 100);
		fprintf(f, "%d,%lf,%lf,%lf,%lf\n", M, tempo_only_host.all, tempo_svm.all, tempo_svmGPU.mult,
				tempo_onlyGPU.mult);
	}
	fclose(f);

	clReleaseCommandQueue(queue);
	WrapperCL_release(&cl);
	return 0;
}

Time_proces svm_only_host(QUEUE q, WrapperCL *cl, int m, int n) {
	LCG rd = new_LCG(SEED);
	Tensor A, B, C;
	Time_proces tempo = {0};
	CNN_ERROR error = {0};
	tempo.init = GetTIME();
	A = new_Tensor(cl->context, q, TENSOR_HMEM | TENSOR_HOST, m, n, 1, 1, &error, NULL);
	B = new_Tensor(cl->context, q, TENSOR_HMEM | TENSOR_HOST, n, n, 1, 1, &error, NULL);
	C = new_Tensor(cl->context, q, TENSOR_HMEM | TENSOR_HOST, m, n, 1, 1, &error, NULL);
	tempo.aux = GetTIME();
	tempo.create_vars = tempo.aux - tempo.init;

	// iniciar  valores
	for (int i = 0; i < A->x; i++) {
		for (int j = 0; j < A->y; j++) {
			Tensor_Map_V(A, i, j, 0) = pLCG_randD(&rd);
		}
	}
	for (int i = 0; i < B->x; i++) {
		for (int j = 0; j < B->y; j++) {
			Tensor_Map_V(B, i, j, 0) = pLCG_randD(&rd);
		}
	}

	tempo.putvalues = GetTIME() - tempo.aux;
	tempo.aux = GetTIME();
	double soma;
	for (int i = 0; i < A->x; i++) {
		for (int j = 0; j < B->y; j++) {
			soma = 0;
			for (int k = 0; k < B->x; k++) {
				soma += Tensor_Map_V(A, i, k, 0) * Tensor_Map_V(B, k, j, 0);
			}
			Tensor_Map_V(C, i, j, 0) = soma;
		}
	}
	tempo.mult = GetTIME() - tempo.aux;
	releaseTensor(&A);
	releaseTensor(&B);
	releaseTensor(&C);
	tempo.all = GetTIME() - tempo.init;
	return tempo;
}

Time_proces svm_using_svm(QUEUE q, WrapperCL *cl, int m, int n) {
	LCG rd = new_LCG(SEED);
	Tensor A, B, C;
	Time_proces tempo = {0};
	CNN_ERROR error = {0};
	tempo.init = GetTIME();
	A = new_Tensor(cl->context, q, TENSOR_SMEM | TENSOR_HOST, m, n, 1, 1, &error, NULL);
	B = new_Tensor(cl->context, q, TENSOR_SMEM | TENSOR_HOST, n, n, 1, 1, &error, NULL);
	C = new_Tensor(cl->context, q, TENSOR_SMEM | TENSOR_HOST, m, n, 1, 1, &error, NULL);
	tempo.aux = GetTIME();
	tempo.create_vars = tempo.aux - tempo.init;

	// iniciar  valores
	for (int i = 0; i < A->x; i++) {
		for (int j = 0; j < A->y; j++) {
			Tensor_Map_V(A, i, j, 0) = pLCG_randD(&rd);
		}
	}
	for (int i = 0; i < B->x; i++) {
		for (int j = 0; j < B->y; j++) {
			Tensor_Map_V(B, i, j, 0) = pLCG_randD(&rd);
		}
	}

	tempo.putvalues = GetTIME() - tempo.aux;
	tempo.aux = GetTIME();
	double soma;
	for (int i = 0; i < A->x; i++) {
		for (int j = 0; j < B->y; j++) {
			soma = 0;
			for (int k = 0; k < B->x; k++) {
				soma += Tensor_Map_V(A, i, k, 0) * Tensor_Map_V(B, k, j, 0);
			}
			Tensor_Map_V(C, i, j, 0) = soma;
		}
	}
	tempo.mult = GetTIME() - tempo.aux;
	releaseTensor(&A);
	releaseTensor(&B);
	releaseTensor(&C);
	tempo.all = GetTIME() - tempo.init;
	return tempo;
}

Time_proces svm_GPU_using_svm(QUEUE q, WrapperCL *cl, int m, int n) {
	LCG rd = new_LCG(SEED);
	Tensor A, B, C;
	Time_proces tempo = {0};
	CNN_ERROR error = {0};
	Kernel kernel_mult = new_Kernel(cl->program, &error, mult, 5, K_VOID_P, K_VOID_P, K_VOID_P, K_INT, K_INT);
	tempo.init = GetTIME();
	A = new_Tensor(cl->context, q, TENSOR_SMEM, m, n, 1, 1, &error, NULL);
	B = new_Tensor(cl->context, q, TENSOR_SMEM, n, n, 1, 1, &error, NULL);
	C = new_Tensor(cl->context, q, TENSOR_SMEM, m, n, 1, 1, &error, NULL);
	if (error.error) {
		showError(error.error);
		goto end;
	}


	tempo.aux = GetTIME();
	tempo.create_vars = tempo.aux - tempo.init;
	// iniciar  valores
	for (int i = 0; i < A->x; i++) {
		for (int j = 0; j < A->y; j++) {
			Tensor_Map_V(A, i, j, 0) = 1;//pLCG_randD(&rd);
		}
	}

	for (int i = 0; i < B->x; i++) {
		for (int j = 0; j < B->y; j++) {
			Tensor_Map_V(B, i, j, 0) = pLCG_randD(&rd);
		}
	}

	tempo.putvalues = GetTIME() - tempo.aux;
	tempo.aux = GetTIME();
	kernel_run_recursive(error.error, kernel_mult, q, m * n, cl->maxworks, &A->data, &B->data, &C->data, &A->y);
	synchronizeKernel(q);
	tempo.mult = GetTIME() - tempo.aux;
	end:
	releaseKernel(&kernel_mult);
	releaseTensor(&A);
	releaseTensor(&B);
	releaseTensor(&C);
	tempo.all = GetTIME() - tempo.init;
	return tempo;
}

Time_proces svm_GPU(QUEUE q, WrapperCL *cl, int m, int n) {
	LCG rd = new_LCG(SEED);
	Tensor A, B, C;
	Time_proces tempo = {0};
	CNN_ERROR error = {0};
	Kernel kernel_mult = new_Kernel(cl->program, &error, mult, 5, K_VOID_P, K_VOID_P, K_VOID_P, K_INT, K_INT);
	tempo.init = GetTIME();
	A = new_Tensor(cl->context, q, 0, m, n, 1, 1, &error, NULL);
	B = new_Tensor(cl->context, q, 0, n, n, 1, 1, &error, NULL);
	C = new_Tensor(cl->context, q, 0, m, n, 1, 1, &error, NULL);
	if (error.error) {
		showError(error.error);
		goto end;
	}


	tempo.aux = GetTIME();
	tempo.create_vars = tempo.aux - tempo.init;
	double *buff = alloc_mem(m * n, sizeof(double));
	// iniciar  valores
	for (int i = 0; i < A->x; i++) {
		for (int j = 0; j < A->y; j++) {
			buff[Tensor_Map(A, i, j, 0)] = 1;//pLCG_randD(&rd);
		}
	}

	TensorPutValues(q, A, buff);
	for (int i = 0; i < B->x; i++) {
		for (int j = 0; j < B->y; j++) {
			buff[Tensor_Map(B, i, j, 0)] = pLCG_randD(&rd);
		}
	}
	TensorPutValues(q, B, buff);


	tempo.putvalues = GetTIME() - tempo.aux;
	tempo.aux = GetTIME();
	kernel_run_recursive(error.error, kernel_mult, q, m * n, cl->maxworks, &A->data, &B->data, &C->data, &A->y);
	synchronizeKernel(q);
	tempo.mult = GetTIME() - tempo.aux;
	end:
	releaseKernel(&kernel_mult);
	releaseTensor(&A);
	releaseTensor(&B);
	releaseTensor(&C);
	free_mem(buff);
	tempo.all = GetTIME() - tempo.init;
	return tempo;
}
