//
// Created by Henrique on 22/05/2021.
//

#include "tensor/Tensor.h"

void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, CNN_ERROR *error, void *p);

Tensor new_Tensor(cl_context context, QUEUE queue, char tensor_flag, UINT x, UINT y, UINT z, UINT w, CNN_ERROR *error,
				  void *p) {
	if (error->error)return NULL;
	Tensor t = (Tensor) alloc_mem(1, sizeof(struct typetensor));
	t->x = x;
	t->y = y;
	t->z = z;
	t->w = w;
	t->flag = tensor_flag;
	switch (tensor_flag & TENSOR_MASK_DIM) {
		case TENSOR3D:
			t->w = 1;
			break;
		case TENSOR4D:
			break;
		default:
			error->error = TENSOR_INVALID_FLAG_DIM;
			free_mem(t);
			return NULL;
	}

	switch (tensor_flag & TENSOR_MASK_TYPE) {
		case TENSOR_DOUBLE:
			t->bytes = x * y * z * sizeof(double);
			break;
		case TENSOR_CHAR:
			t->bytes = x * y * z * sizeof(char);
			break;
		case TENSOR_INT:
			t->bytes = x * y * z * sizeof(int);
			break;
		default:
			error->error = TENSOR_INVALID_FLAG_TYPE;
			free_mem(t);
			return NULL;

	}
	__fillTensor__(t, context, queue, t->bytes * t->w, error, p);
	if (error->error) {
		releaseTensor(&t);
	}
	return t;
}


void releaseTensor(Tensor *t) {
	if (*t) {
		if (((*t)->flag & TENSOR_MASK_DRIVEORHOST) == TENSOR_HOST) {
			// apenas no host
			switch ((*t)->flag & TENSOR_MASK_MEM) {
				case TENSOR_UPTR:
					break;
				case TENSOR_SMEM:
					if ((*t)->host)
						free_cl_svm((*t)->context, (*t)->host);
					(*t)->data = NULL;
					(*t)->host = NULL;
					break;
				case TENSOR_HMEM:
					if ((*t)->host)
						free_mem((*t)->host);
					(*t)->data = NULL;
					(*t)->host = NULL;
					break;
				default:
					fprintf(stderr, "invalid flag tensor\n");
					break;
			}

		} else {
			// use gpu
			if ((*t)->data) {
				clReleaseMemObject((*t)->data);
			}
			(*t)->data = NULL;
			switch ((*t)->flag & TENSOR_MASK_MEM) {
				case TENSOR_SMEM:
					if ((*t)->host)
						free_cl_svm((*t)->context, (*t)->host);
					(*t)->host = NULL;
					break;
				case TENSOR_HMEM:
					if ((*t)->host)
						free_mem((*t)->host);

					(*t)->host = NULL;
					break;
				case TENSOR_NCPY:
				case TENSOR_UPTR:
					break;
				default:
					fprintf(stderr, "invalid flag tensor\n");
					break;
			}
		}
		free_mem(*t);
		*t = NULL;
	}
}


void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, CNN_ERROR *error, void *p) {
	if (error->error)return;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "__fillTensor__");
#if  (RUN_KERNEL_USING_GPU != 1)
	t->flag = TENSOR_HOST;
#warning runing kernels into host
#endif//RUN_KERNEL_USING_GPU
	if ((t->flag & TENSOR_MASK_DRIVEORHOST) == TENSOR_HOST) {
		// rodando no host
		switch (t->flag & TENSOR_MASK_MEM) {
			case TENSOR_UPTR:
				if (!p) {
					error->error = NULL_PARAM;
					releaseTensor(&t);
				}
				t->data = t->host = p;
				break;
			case TENSOR_HMEM:
				t->data = t->host = alloc_mem(bytes, 1);
				break;
			case TENSOR_SMEM:
				t->context = context;
				t->host = alloc_cl_svm(t->context, CL_MEM_READ_WRITE, bytes, 0);
				t->data = t->host;
				break;
			default:
				error->error = TENSOR_INVALID_FLAG_MEM;
				releaseTensor(&t);
				return;
		}
		return;
	}
	// rodando no drive
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_UPTR:
			if (!p) {
				error->error = NULL_PARAM;
				releaseTensor(&t);
			}
			t->host = p;
			t->data = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, bytes, t->host, &error->error);
			break;
		case TENSOR_SMEM:
			t->context = context;
			t->host = alloc_cl_svm(t->context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, bytes, 0);
			t->data = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes, t->host, &error->error);

			break;
		case TENSOR_NCPY:
			t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &error->error);
			break;
		default:
			error->error = TENSOR_INVALID_FLAG_MEM;
			releaseTensor(&t);
			return;
	}
}


int TensorFill(QUEUE queue, Tensor t, char pattern) {
	return TensorFillOffSet(queue, t, pattern, 0);
}

int TensorFillOffSet(QUEUE queue, Tensor t, char pattern, size_t offset) {
	int erro = 0;
	void *mem;
	if ((t->flag & TENSOR_MASK_DRIVEORHOST) == TENSOR_HOST) {
		memset(t->host + offset, pattern, t->bytes);
		return erro;
	}
	switch (t->flag & TENSOR_MASK_MEM) {

		case TENSOR_HMEM:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, offset, t->bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorFillOffSet/clEnqueueMapBuffer ");
			memset(mem + offset, pattern, t->bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorFillOffSet/clEnqueueUnmapMemObject ");
			return erro;
		case TENSOR_SMEM:
			synchronizeKernel(queue);
			mem = t->host;
			memset(mem + offset, pattern, t->bytes);
			return erro;
		case TENSOR_UPTR:
		case TENSOR_NCPY:
			erro = clEnqueueFillBuffer(queue, t->data, &pattern, sizeof(char), offset, t->bytes, 0, NULL, NULL);
			PERR(erro, "TensorFillOffSet/clEnqueueWriteBuffer ");
			return erro;

		default:
			erro = TENSOR_INVALID_FLAG_MEM;
			PERR(erro, "TensorFillOffSet: INVALID TENSOR FLAG %d ", t->flag);

	}
}

int TensorFillDouble(QUEUE queue, Tensor t, double pattern) {
	return TensorFillDoubleOffSet(queue, t, pattern, 0);
}

int TensorFillDoubleOffSet(QUEUE queue, Tensor t, double pattern, size_t offset) {
	int erro = 0;
	double *mem;
	if ((t->flag & TENSOR_MASK_DRIVEORHOST) == TENSOR_HOST) {
		mem = t->host;
		for (int i = t->x * t->y * t->z * t->w - 1; i >= 0; i--) {
			mem[i] = pattern;
		}
		return erro;
	}
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_HMEM:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, offset, t->bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorFillOffSet/clEnqueueMapBuffer ");
			for (int i = t->x * t->y * t->z * t->w - 1; i >= 0; i--) {
				mem[i] = pattern;
			}
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorFillOffSet/clEnqueueUnmapMemObject ");
			return erro;
		case TENSOR_SMEM:
			synchronizeKernel(queue);
			mem = t->host;
			for (int i = t->x * t->y * t->z * t->w - 1; i >= 0; i--) {
				mem[i] = pattern;
			}
			return erro;
		case TENSOR_UPTR:
		case TENSOR_NCPY:
			erro = clEnqueueFillBuffer(queue, t->data, &pattern, sizeof(double), offset, t->bytes, 0, NULL, NULL);
			PERR(erro, "TensorFillOffSet/clEnqueueWriteBuffer ");
			return erro;

		default:
			erro = TENSOR_INVALID_FLAG_MEM;
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
	if ((t->flag & TENSOR_MASK_DRIVEORHOST) == TENSOR_HOST) {
		memcpy(data, t->host + offset, bytes);
		return erro;
	}
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_HMEM:
			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, offset, bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueMapBuffer");
			memcpy(data, mem, bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueUnmapMemObject");
			return erro;
		case TENSOR_SMEM:
			synchronizeKernel(queue);
			mem = t->host;
			memcpy(data, mem, bytes);
			return erro;
		case TENSOR_UPTR:
		case TENSOR_NCPY:
			erro = clEnqueueReadBuffer(queue, t->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueReadBuffer %d 0x%p 0x%p", (int) t->flag, t->data, data);
			return erro;

		default:
			erro = TENSOR_INVALID_FLAG_MEM;
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
	if ((t->flag & TENSOR_MASK_DRIVEORHOST) == TENSOR_HOST) {
		memcpy(t->host + ofset, data, bytes);
		return erro;
	}
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_HMEM:

			mem = clEnqueueMapBuffer(queue, t->data, CL_TRUE, CL_MAP_WRITE, ofset, bytes, 0, 0, 0, &erro);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueMapBuffer ");
			memcpy(mem, data, bytes);
			erro = clEnqueueUnmapMemObject(queue, t->data, mem, 0, 0, 0);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueUnmapMemObject ");
			return erro;
		case TENSOR_SMEM:
			synchronizeKernel(queue);
			mem = t->host;
			memcpy(mem, data, bytes);
			return erro;
		case TENSOR_UPTR:
		case TENSOR_NCPY:
			erro = clEnqueueWriteBuffer(queue, t->data, CL_TRUE, ofset, bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorPutValuesOffSet/clEnqueueWriteBuffer ");
			return erro;

		default:
			erro = TENSOR_INVALID_FLAG_MEM;
			PERR(erro, "TensorPutValuesOffSet: INVALID TENSOR FLAG %d ", t->flag);
	}

}

int dividirVetor(double *v, Tensor m, size_t len, double value, Kernel funcNorm, size_t max_works,
                 QUEUE queue) {
	int error = TensorPutValuesMem(queue, m, v, len * sizeof(double));
	if (error)return error;
	kernel_run_recursive(error, funcNorm, queue, len, max_works, K_ARG m, K_ARG value);
	if (error)return error;
	error = TensorGetValuesMem(queue, m, v, len * sizeof(double));
	return error;
}


void printTensor(QUEUE q, Tensor t, FILE *f) {
	union {
		double *d;
		int *i;
		unsigned char *c;
	} v;
	v.d = alloc_mem(t->bytes, 1);
	char buff[EXCEPTION_MAX_MSG_SIZE];
	int error = 0;
	fprintf(f, "%u %u %u %u (%s)\n", t->x, t->y, t->z, t->w, printBytes(t->bytes * t->w, buff));
	for (int l = 0; l < t->w; l++) {
		error = TensorGetValuesOffSet(q, t, v.d, l * t->bytes);
		if (error)break;
		for (int z = 0; z < t->z; z++) {
			for (int i = 0; i < t->x; i++) {
				fprintf(f, "  ");
				for (int j = 0; j < t->y; j++)
					switch (t->flag & TENSOR_MASK_TYPE) {
						case TENSOR_DOUBLE:
							fprintf(f, "%.2g ", v.d[Tensor_Map(t, i, j, z)]);
							break;
						case TENSOR_INT:
							fprintf(f, "%d ", v.i[Tensor_Map(t, i, j, z)]);
							break;
						case TENSOR_CHAR:
							fprintf(f, "%2X ", (int) v.c[Tensor_Map(t, i, j, z)]);
							break;
					default:
						free_mem(v.d);

					}

				fprintf(f, "\n");
			}
			if (z + 1 != t->z)
				fprintf(f, "\n");
		}
		if (l + 1 != t->w)
			fprintf(f, "------------\n");
	}
	fprintf(f, "\n");
	free_mem(v.d);
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
	double *v = alloc_mem(t->bytes, t->w);
	int a;
	for (a = 0; a < t->w && !error; ++a) {
		error = TensorGetValuesOffSet(queue, t, v + a, a * t->bytes);
	}
	if (error) {
		fprintf(stderr, "TensorGetNorm/TensorGetValuesOffSet(%d of %d):", a, t->w);
		showError(error);
		free_mem(v);
		return error;
	}
	double sum = 0;
	for (int i = t->x * t->y * t->z * t->w - 1; i >= 0; i--) {
		sum += v[i] * v[i];
	}
	*norm = sqrt(sum);
	free_mem(v);
	return error;
}


int TensorAt(Tensor t, UINT x, UINT y, UINT z, UINT w, UINT *index) {
	int erro = 0;
	UINT ofset = y + x * t->y + z * t->x * t->y + w * t->z * t->x * t->y;

	switch (t->flag & TENSOR_MASK_TYPE) {
		case TENSOR_DOUBLE:
			*index = ofset * sizeof(double);
			break;
		case TENSOR_INT:
			*index = ofset * sizeof(int);
			break;
		case TENSOR_CHAR:
			*index = ofset * sizeof(char);
			break;
		default:
			erro = TENSOR_INVALID_FLAG_TYPE;
			PERR(erro, "TensorPutValuesOffSet: INVALID TENSOR FLAG %d ", t->flag);
	}

	return erro;
}

