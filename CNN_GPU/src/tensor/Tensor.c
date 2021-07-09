//
// Created by petel on 22/05/2021.
//

#include "Tensor.h"

char CNN_FILL_TENSOR[] = "FILL_TENSOR";
char CNN_NEW_TENSOR[] = "NEW_TENSOR";
size_t allmem = 0;

void fillTensor(Tensor t, cl_context context, QUEUE queue, size_t bytes, GPU_ERROR *error) {
	if (error->error)return;
	error->context = CNN_FILL_TENSOR;
	int flag = 0;

	if (t->flag) {
		flag = CL_MEM_USE_HOST_PTR;
		t->host = calloc(bytes, 1);
		((double *) t->host)[0] = 10;
	}
	t->data = clCreateBuffer(context, flag | CL_MEM_READ_WRITE, bytes, t->host, &error->error);
	if (!t->data) {
		error->error = -79;
		snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
	}
	if (error->error) {
		snprintf(error->msg, 255, "nao foi possivel allocar memoria vram\n");
	}


	if (!error->error && !t->flag) {
		flag = 0;
		error->error = clEnqueueFillBuffer(queue, t->data, &flag, sizeof(char), 0, bytes, 0, NULL, NULL);
		if (error->error) {
			getClError(error->error, error->msg);
		}
	}
	char buf[250];
	allmem += bytes;
//	printf("%d ",t->flag);
//	printf("buffer alocado de %s\n\t", printBytes(bytes, buf));
//	printf("total usado %s\n", printBytes(allmem, buf));
}


int TensorPutValues(QUEUE queue, Tensor t, void *data) {
	return TensorPutValuesOffSet(queue, t, data, 0);
}

int TensorPutValuesOffSet(QUEUE queue, Tensor t, void *data, UINT ofset) {
	if (!t->flag)
		return clEnqueueWriteBuffer(queue, t->data, CL_TRUE, ofset, t->bytes, data, 0, NULL, NULL);
	int erro = 0;
	void *mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, ofset, t->bytes, 0, 0, 0, &erro);
	if (erro)return erro;
	memcpy(mem, data, t->bytes);
	erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
	return erro;

}


int TensorGetValues(QUEUE queue, Tensor t, void *data) {
	return TensorGetValuesOffset(queue, t, data, 0);
}

int TensorGetValuesOffset(QUEUE queue, Tensor t, void *data, int offset) {
	if (!t->flag) return clEnqueueReadBuffer(queue, t->data, CL_TRUE, offset, t->bytes, data, 0, NULL, NULL);
	int erro = 0;
	void *mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_READ, offset, t->bytes, 0, 0, 0, &erro);
	printf("0x%p 0x%p 0x%p\n", mem, t->host, t->data);
	if (erro)return erro;
	memcpy(data, mem, t->bytes);
	erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
	return erro;
}


Tensor newTensor(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char usehost, GPU_ERROR *error) {
	if (error->error)return NULL;
//	error->context = CNN_NEW_TENSOR;
	if (x <= 0 | y <= 0 | z <= 0) {
		error->error = -1;
	}
	Tensor t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(double);
	t->x = x;
	t->y = y;
	t->z = z;
	t->l = 1;
	t->flag = usehost;
	fillTensor(t, context, queue, t->bytes, error);
	return t;
}


Tensor newTensor4D(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, UINT l, char usehost, GPU_ERROR *error) {
	if (error->error)return NULL;
	error->context = CNN_NEW_TENSOR;
	Tensor t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(double);
	t->x = x;
	t->y = y;
	t->z = z;
	t->l = l;
	fillTensor(t, context, queue, t->bytes * l, error);
	return t;
}


TensorChar newTensorChar(cl_context context, QUEUE queue, UINT x, UINT y, UINT z, char usehost, GPU_ERROR *error) {
	if (error->error)return NULL;
	error->context = CNN_NEW_TENSOR;
	TensorChar t = (Tensor) calloc(1, sizeof(typetensor));
	t->bytes = x * y * z * sizeof(char);
	t->x = x;
	t->y = y;
	t->z = z;
	t->l = 1;
	fillTensor(t, context, queue, t->bytes, error);
	return t;
}


void releaseTensor(Tensor *t) {
	if (*t) {
		LOG_CNN_TENSOR_MEMORY("free (0x%X,0x%X)", *t, (*t)->data)
		if ((*t)->data)
			clReleaseMemObject((*t)->data);
		if ((*t)->host)
			free((*t)->host);
		free(*t);
		*t = NULL;
	}
}


void releaseTensorChar(TensorChar *t) {
	releaseTensor(t);
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


void dividirVetor(double *v, cl_mem m, size_t len, double value, Kernel funcNorm, size_t max_works,
                  QUEUE queue) {
	int error = 0, id = 0;
	size_t global, local, resto;
	clEnqueueWriteBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
	kernel_run_recursive(&funcNorm, queue, len, max_works, &m, &value);
	clFinish(queue);
	clEnqueueReadBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
}


void dividirVetorInt(unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, double value,
                     Kernel funcNorm, size_t max_works, QUEUE queue) {
	clEnqueueWriteBuffer(queue, mi, CL_TRUE, 0, len * sizeof(unsigned char), src, 0, NULL, NULL);
	kernel_run_recursive(&funcNorm, queue, len, max_works, &mi, &mout, &value);
	clFinish(queue);
	clEnqueueReadBuffer(queue, mout, CL_TRUE, 0, len * sizeof(double), dst, 0, NULL, NULL);
}


void int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, int nop,
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

	clEnqueueWriteBuffer(queue, mi, CL_TRUE, 0, len * sizeof(unsigned char), src, 0, NULL, NULL);
	kernel_run_recursive(&func, queue, len, cl->maxworks, &mi, &mout, &nop);
	clFinish(queue);
	clEnqueueReadBuffer(queue, mout, CL_TRUE, 0, len * nop * sizeof(double), dst, 0, NULL, NULL);
}

void printTensor(QUEUE q, Tensor t, FILE *f) {
	double *v = calloc(t->bytes, 1);
	char buff[50];
	fprintf(f, "%u %u %u %u (%s)\n", t->x, t->y, t->z, t->l, printBytes(t->bytes * t->l, buff));
	for (int l = 0; l < t->l; l++) {
		TensorGetValuesOffset(q, t, v, l * t->bytes);
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

}

