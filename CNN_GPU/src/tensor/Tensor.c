//
// Created by Henrique on 22/05/2021.
//

#include "Tensor.h"

void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, Exception *error);

Tensor newTensor(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char usehost, Exception *error) {
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
	t->flag = usehost;
	__fillTensor__(t, context, queue, t->bytes, error);

	return t;
}


Tensor newTensor4D(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, UINT l, char usehost, Exception *error) {
	if (error->error)return NULL;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "newTensor4D");
	Tensor t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(double);
	t->x = x;
	t->y = y;
	t->z = z;
	t->w = l;
	__fillTensor__(t, context, queue, t->bytes * l, error);

	return t;
}


TensorChar newTensorChar(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char usehost, Exception *error) {
	if (error->error)return NULL;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "newTensorChar");
	TensorChar t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(char);
	t->x = x;
	t->y = y;
	t->z = z;
	t->w = 1;
	__fillTensor__(t, context, queue, t->bytes, error);

	return t;
}


void releaseTensor(Tensor *t) {
	if (*t) {
		LOG_CNN_TENSOR_MEMORY("free (0x%X,0x%X)", *t, (*t)->data)
		switch ((*t)->flag) {
			case TENSOR_HOST:
				if ((*t)->host)
					free((*t)->host);
			case TENSOR_HSTA:
			case TENSOR_NCPY:
				if ((*t)->data)
					clReleaseMemObject((*t)->data);
				break;
			case TENSOR_SVMA:
			case TENSOR_SVM:
				if ((*t)->host)
					clSVMFree((*t)->context, (*t)->host);
				break;
		}
		*t = NULL;
	}
}


void releaseTensorChar(TensorChar *t) {
	releaseTensor(t);
}

size_t allmem = 0;


void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, Exception *error) {
	if (error->error)return;
	t->context = context;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "__fillTensor__");
	switch (t->flag) {
		case TENSOR_HOST:
			t->host = calloc(bytes, 1);
			t->data = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes, t->host, &error->error);
			break;
		case TENSOR_HSTA:
			t->data = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, bytes, &t->host,
			                         &error->error);
			break;
		case TENSOR_NCPY:
			t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &error->error);
			break;
		case TENSOR_SVM:
			t->host = clSVMAlloc(context, CL_MEM_READ_WRITE, bytes, 0);
			t->data = t->host;
			break;
		case TENSOR_SVMA:
			t->host = clSVMAlloc(context, CL_MEM_SVM_ATOMICS | CL_MEM_READ_WRITE, bytes, 0);
			t->data = t->host;
			break;
	}

	if (!t->data) {
		error->error = -79;
		snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
	}
	if (error->error) {
		getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
	}
	if (error->error) return;
//	char buf[250];
	allmem += bytes;
//	printf("%d ",t->flag);
//	printf("buffer alocado de %s\n\t", printBytes(bytes, buf));
//	printf("total usado %s\n", printBytes(allmem, buf));
}

int TensorFill(QUEUE queue, Tensor t, char pattern) {
	return TensorFillOffSet(queue, t, pattern, 0);
}

int TensorFillOffSet(QUEUE queue, Tensor t, char pattern, UINT offset) {
	int erro = 0;
	void *mem;
	switch (t->flag) {
		case TENSOR_HOST:
		case TENSOR_HSTA:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, offset, t->bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorFillOffSet/clEnqueueMapBuffer ");
			memset(mem, pattern, t->bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorFillOffSet/clEnqueueUnmapMemObject ");
			return erro;

		case TENSOR_NCPY:
			erro = clEnqueueFillBuffer(queue, t->data, &pattern, sizeof(char), offset, t->bytes, 0, NULL, NULL);
			PERR(erro, "TensorFillOffSet/clEnqueueWriteBuffer ");
			return erro;
		case TENSOR_SVM:
		case TENSOR_SVMA:
			erro = clFinish(queue);
			PERR(erro, "TensorFillOffSet/clFinish ");
			memset(t->host, pattern, t->bytes);
			return erro;
		default:
			erro = -90;
			PERR(erro, "TensorFillOffSet: INVALID TENSOR FLAG %d ", t->flag);

	}
}

int TensorPutValues(QUEUE queue, Tensor t, void *data) {
	return TensorPutValuesOffSet(queue, t, data, 0);
}


int TensorPutValuesOffSet(QUEUE queue, Tensor t, void *data, UINT ofset) {
	int erro = 0;
	void *mem;
	switch (t->flag) {
		case TENSOR_HOST:
		case TENSOR_HSTA:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, ofset, t->bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueMapBuffer ");
			memcpy(mem, data, t->bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueUnmapMemObject ");
			return erro;

		case TENSOR_NCPY:
			erro = clEnqueueWriteBuffer(queue, t->data, CL_TRUE, ofset, t->bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueWriteBuffer ");

			return erro;
		case TENSOR_SVM:
		case TENSOR_SVMA:
			erro = clFinish(queue);
			PERR(erro, "TensorPutValuesOffSet/clFinish ");
			memcpy(t->host, data, t->bytes);
			return erro;
		default:
			erro = -90;
			PERR(erro, "TensorPutValuesOffSet: INVALID TENSOR FLAG %d ", t->flag);

	}

}

int TensorGetValues(QUEUE queue, Tensor t, void *data) {
	return TensorGetValuesOffset(queue, t, data, 0);
}


int TensorGetValuesOffset(QUEUE queue, Tensor t, void *data, unsigned int offset) {
	int erro;
	void *mem;
	switch (t->flag) {
		case TENSOR_HOST:
		case TENSOR_HSTA:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, offset, t->bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorGetValuesOffset/clEnqueueMapBuffer");
			memcpy(data, mem, t->bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorGetValuesOffset/clEnqueueUnmapMemObject");
			return erro;

		case TENSOR_NCPY:
			erro = clEnqueueReadBuffer(queue, t->data, CL_TRUE, offset, t->bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorGetValuesOffset/clEnqueueReadBuffer");
			return erro;
		case TENSOR_SVM:
		case TENSOR_SVMA:
			erro = clFinish(queue);
			PERR(erro, "TensorGetValuesOffset/clFinish ");
			memcpy(data, t->host, t->bytes);
			return erro;
		default:
			erro = -90;
			PERR(erro, "TensorGetValuesOffset: INVALID TENSOR FLAG %d ", t->flag);
	}
}


TensorC newTensorC(int x, int y, int z) {
	TensorC t = (TensorC) calloc(1, sizeof(typeTensorC));
	t->x = x;
	t->y = y;
	t->z = z;
	t->data = (double *) calloc(x * y * z, sizeof(double));
	LOG_CNN_TENSOR_MEMORY("aloc CTENSOR (0x%X,0x%X)", t, t->data)
	return t;
}

void releaseTensorC(TensorC c) {
	if (c) {
		LOG_CNN_TENSOR_MEMORY("free CTENSOR (0x%X,0x%X)", *t, (*t)->data)
		free(c->data);
		free(c);
	}
}


int dividirVetor(double *v, cl_mem m, size_t len, double value, Kernel funcNorm, size_t max_works,
                 QUEUE queue) {
	int error = clEnqueueWriteBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
	if (error)return error;
	error = kernel_run_recursive(&funcNorm, queue, len, max_works, &m, &value);
	if (error)return error;
	error = clEnqueueReadBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
	return error;
}


int dividirVetorInt(unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, double value,
                    Kernel funcNorm, size_t max_works, QUEUE queue) {
	int error = clEnqueueWriteBuffer(queue, mi, CL_TRUE, 0, len * sizeof(unsigned char), src, 0, NULL, NULL);
	if (error)return error;
	error = kernel_run_recursive(&funcNorm, queue, len, max_works, &mi, &mout, &value);
	if (error)return error;
	error = clEnqueueReadBuffer(queue, mout, CL_TRUE, 0, len * sizeof(double), dst, 0, NULL, NULL);
	return error;
}


int int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, int nop,
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

	int error = clEnqueueWriteBuffer(queue, mi, CL_TRUE, 0, len * sizeof(unsigned char), src, 0, NULL, NULL);
	if (error)return error;
	error = kernel_run_recursive(&func, queue, len, cl->maxworks, &mi, &mout, &nop);
	if (error)return error;
	error = clEnqueueReadBuffer(queue, mout, CL_TRUE, 0, len * nop * sizeof(double), dst, 0, NULL, NULL);
	return error;
}

void printTensor(QUEUE q, Tensor t, FILE *f) {
	double *v = calloc(t->bytes, 1);
	char buff[EXCEPTION_MAX_MSG_SIZE];
	int error = 0;
	fprintf(f, "%u %u %u %u (%s)\n", t->x, t->y, t->z, t->w, printBytes(t->bytes * t->w, buff));
	for (int l = 0; l < t->w; l++) {
		error = TensorGetValuesOffset(q, t, v, l * t->bytes);
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

