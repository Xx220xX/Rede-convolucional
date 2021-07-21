//
// Created by Henrique on 22/05/2021.
//

#include "Tensor.h"

void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, Exception *error);

Tensor newTensor(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char tensor_flag, Exception *error) {
	if (error->error)return NULL;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "newTensor");
	if (x <= 0 | y <= 0 | z <= 0) {
		error->error = -1;
	}
	Tensor t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(double);
	t->x = x;
	t->y = y;
	t->z = z;
	t->w = 1;
#ifndef DISABLE_KERNELS_INSIDE_DRIVE
	t->flag = tensor_flag;
#else
	t->flag = TENSOR_HOST;
#endif//DISABLE_KERNELS_INSIDE_DRIVE
	__fillTensor__(t, context, queue, t->bytes, error);
	return t;
}


Tensor newTensor4D(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, UINT l, char tensor_flag, Exception *error) {
	if (error->error)return NULL;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "newTensor4D");
	Tensor t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(double);
	t->x = x;
	t->y = y;
	t->z = z;
	t->w = l;
	t->flag = tensor_flag;
	__fillTensor__(t, context, queue, t->bytes * l, error);

	return t;
}


TensorChar newTensorChar(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char tensor_flag, Exception *error) {
	if (error->error)return NULL;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "newTensorChar");
	TensorChar t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(char);
	t->x = x;
	t->y = y;
	t->z = z;
	t->w = 1;
	t->flag = tensor_flag;
	__fillTensor__(t, context, queue, t->bytes, error);

	return t;
}


void releaseTensor(Tensor *t) {
	if (*t) {
		switch ((*t)->flag) {
			case TENSOR_HOST:
				if ((*t)->host)
					free((*t)->host);
				break;
			case TENSOR_SMEM:
				clSVMFree((*t)->context, (*t)->host);
			case TENSOR_NCPY:
				if ((*t)->data)
					clReleaseMemObject((*t)->data);
				break;

		}
		*t = NULL;
	}
}


void releaseTensorChar(TensorChar *t) {
	releaseTensor(t);
}


void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, Exception *error) {
	if (error->error)return;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "__fillTensor__");
	switch (t->flag) {
		case TENSOR_HOST:
			t->host = calloc(bytes, 1);
			t->data = t->host;
			break;
		case TENSOR_SMEM:
			t->context = context;
			t->host = clSVMAlloc(t->context, CL_MEM_READ_WRITE, bytes, 0);
			t->data = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes, t->host, &error->error);
			break;
		case TENSOR_NCPY:
			t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &error->error);
			break;
	}


	if (error->error) {
		getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
		return;
	}
	if (!t->data) {
		error->error = -79;
		snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
	}
	if (error->error) return;
}

int TensorFill(QUEUE queue, Tensor t, char pattern) {
	return TensorFillOffSet(queue, t, pattern, 0);
}

int TensorFillOffSet(QUEUE queue, Tensor t, char pattern, size_t offset) {
	int erro = 0;
	void *mem;
	switch (t->flag) {
		case TENSOR_HOST:
			memset(t->host + offset, pattern, t->bytes);
			return erro;
		case TENSOR_SMEM:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, offset, t->bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorFillOffSet/clEnqueueMapBuffer ");
			memset(mem + offset, pattern, t->bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorFillOffSet/clEnqueueUnmapMemObject ");
			return erro;

		case TENSOR_NCPY:
			erro = clEnqueueFillBuffer(queue, t->data, &pattern, sizeof(char), offset, t->bytes, 0, NULL, NULL);
			PERR(erro, "TensorFillOffSet/clEnqueueWriteBuffer ");
			return erro;

		default:
			erro = -90;
			PERR(erro, "TensorFillOffSet: INVALID TENSOR FLAG %d ", t->flag);

	}
}


int TensorGetValues(QUEUE queue, Tensor t, void *data) {
	return TensorGetValuesMemOffSet(queue, t, data, t->bytes, 0);
}

int TensorGetValuesMem(QUEUE queue, Tensor t, void *data, size_t bytes) {
	return TensorGetValuesMemOffSet(queue, t, data, bytes, 0);
}

int TensorGetValuesOffSet(QUEUE queue, Tensor t, void *data, size_t offset) {
	return TensorGetValuesMemOffSet(queue, t, data, t->bytes, offset);
}

int TensorGetValuesMemOffSet(QUEUE queue, Tensor t, void *data, size_t bytes, size_t offset) {
	int erro = 0;
	void *mem;
	switch (t->flag) {
		case TENSOR_HOST:
			memcpy(data, t->host, bytes);
			return erro;
		case TENSOR_SMEM:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, offset, bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueMapBuffer");
			memcpy(data, mem, bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueUnmapMemObject");
			return erro;

		case TENSOR_NCPY:
			erro = clEnqueueReadBuffer(queue, t->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueReadBuffer %d 0x%p 0x%p", (int) t->flag, t->data, data);
			return erro;

		default:
			erro = -90;
			PERR(erro, "TensorGetValuesOffSet: INVALID TENSOR FLAG %d ", t->flag);
	}
}

int TensorPutValues(QUEUE queue, Tensor t, void *data) {
	return TensorPutValuesMemOffSet(queue, t, data, t->bytes, 0);
}


int TensorPutValuesOffSet(QUEUE queue, Tensor t, void *data, size_t ofset) {
	return TensorPutValuesMemOffSet(queue, t, data, t->bytes, ofset);
}


int TensorPutValuesMem(QUEUE queue, Tensor t, void *data, size_t bytes) {
	return TensorPutValuesMemOffSet(queue, t, data, bytes, 0);
}

int TensorPutValuesMemOffSet(QUEUE queue, Tensor t, void *data, size_t bytes, size_t ofset) {
	int erro = 0;
	void *mem;
	switch (t->flag) {
		case TENSOR_HOST:
			memcpy(t->host, data, bytes);
			return erro;
		case TENSOR_SMEM:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, ofset, bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueMapBuffer ");
			memcpy(mem, data, bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueUnmapMemObject ");
			return erro;

		case TENSOR_NCPY:
			erro = clEnqueueWriteBuffer(queue, t->data, CL_TRUE, ofset, bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueWriteBuffer ");

			return erro;

		default:
			erro = -90;
			PERR(erro, "TensorPutValuesOffSet: INVALID TENSOR FLAG %d ", t->flag);

	}

}

int dividirVetor(double *v, Tensor m, size_t len, double value, Kernel funcNorm, size_t max_works,
                 QUEUE queue) {
	int error = TensorPutValuesMem(queue,m,v, len * sizeof(double));
	if (error)return error;
	kernel_run_recursive(error, funcNorm, queue, len, max_works, K_ARG m, K_ARG value);
	if (error)return error;
	error = TensorGetValuesMem(queue,m,v, len * sizeof(double));
	return error;
}


int dividirVetorInt(unsigned char *src, double *dst, Tensor mi, Tensor mout, size_t len, double value,
                    Kernel funcNorm, size_t max_works, QUEUE queue) {
	int error = TensorPutValuesMem(queue, mi,src, len * sizeof(unsigned char));
	if (error)return error;
	kernel_run_recursive(error, funcNorm, queue, len, max_works, K_ARG mi, K_ARG mout, K_ARG value);
	if (error)return error;
	error = TensorGetValuesMem(queue, mout,dst, len * sizeof(double));
	return error;
}


int int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, Tensor mi, Tensor mout, size_t len, int nop,
                     Kernel func, QUEUE queue) {
	/* cl_program pg = compileProgram(cl->context,cl->device,
									 "__kernel void printI(__global unsigned char *v, int len){"
									 "for(int i = 0;i<len;i++){"
									 "printf(\"%d \",v[i]);}printf(\"\\n\");}\n"
									 "__kernel void printD(__global double *v, int len,int len2){"
									 "for(int i = 0;i<len;i++){"
									 "printf(\"[\");"
									 "for(int j = 0;j<len2;j++){printf(\"%.4lf \",v[i*len2+j]);}"
									 "printf(\"]\\n\");}"
									 "printf(\"\\n\");}\n");
	  Kernel printI = new_Kernel(pg,"printI",2,VOID_P,INT);
	  Kernel printD = new_Kernel(pg,"printD",3,VOID_P,INT,INT);
  */

	int error = TensorPutValuesMem(queue, mi,src, len * sizeof(unsigned char));
	if (error)return error;
	kernel_run_recursive(error,func, queue, len, cl->maxworks,K_ARG mi, K_ARG mout, K_ARG nop);
	if (error)return error;
	error = TensorGetValuesMem(queue, mout, dst, len * nop * sizeof(double));
	return error;
}

void printTensor(QUEUE q, Tensor t, FILE *f) {
	double *v = calloc(t->bytes, 1);
	char buff[EXCEPTION_MAX_MSG_SIZE];
	int error = 0;
	fprintf(f, "%u %u %u %u (%s)\n", t->x, t->y, t->z, t->w, printBytes(t->bytes * t->w, buff));
	for (int l = 0; l < t->w; l++) {
		error = TensorGetValuesOffSet(q, t, v, l * t->bytes);
		if (error)break;
		for (int z = 0; z < t->z; z++) {
			fprintf(f, "(:,:,%d,%d)\n", z, l);
			for (int i = 0; i < t->x; i++) {
				for (int j = 0; j < t->y; j++)
					fprintf(f, "%.2g ", v[TensorMap(t, i, j, z)]);
				fprintf(f, "\n");
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	fprintf(f, "\n");
	free(v);
	if (error) {
		fprintf(stderr, "printTensor: %d %s", error, getClError(error, buff, EXCEPTION_MAX_MSG_SIZE));
	}

}

int TensorGetNorm(QUEUE queue, Tensor t, double *norm) {
	int error = 0;
	if (!norm)return -92;
	if (!t) {
		return -93;
	}
	double *v = calloc(t->bytes, t->w);
	int a;
	for (a = 0; a < t->w && !error; ++a) {
		error = TensorGetValuesOffSet(queue, t, v + a, a * t->bytes);
	}
	if (error) {
		fprintf(stderr, "TensorGetNorm/TensorGetValuesOffSet(%d of %d):", a, t->w);
		showError(error);
		free(v);
		return error;
	}
	double sum = 0;
	for (int i = t->x * t->y * t->z * t->w - 1; i >= 0; i--) {
		sum += v[i] * v[i];
	}
	*norm = sqrt(sum);
	free(v);
	return error;
}

