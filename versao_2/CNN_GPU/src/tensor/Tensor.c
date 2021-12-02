//
// Created by Henrique on 22/05/2021.
//

#include "tensor/Tensor.h"

void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, CNN_ERROR *error, void *p);

Tensor new_Tensor(cl_context context, QUEUE queue, char tensor_flag, UINT x, UINT y, UINT z, UINT w, CNN_ERROR *error,
				  void *p) {
	if (error->error)return NULL;
	Tensor t = (Tensor) alloc_mem(1, sizeof(struct Tensor_t));
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
		case TENSOR_REAL:
			t->bytes = x * y * z * sizeof(REAL);
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
		switch ((*t)->flag & TENSOR_MASK_MEM) {
			case TENSOR_RAM:
				free_mem((*t)->host);
				break;
			case TENSOR_SVM:
				free_cl_svm((*t)->context, (*t)->host);
				break;
			case TENSOR_GPU:
				clReleaseMemObject((*t)->data);
				break;
			default:
				fprintf(stderr, "invalid flag tensor\n");
				break;

		}
		free_mem(*t);
		*t = NULL;
	}
}


void __fillTensor__(Tensor t, cl_context context, QUEUE queue, size_t bytes, CNN_ERROR *error, void *p) {
	if (error->error)return;
	//int lencontext = sprintf(error->context + strlen(error->context), "/%s", "__fillTensor__");

#if  (RUN_KERNEL_USING_GPU != 1)
	int flag = t->flag;
	flag = flag * (!TENSOR_MASK_MEM);
	flag |= TENSOR_RAM;
	t->flag = flag;
#warning runing kernels into host
#endif//RUN_KERNEL_USING_GPU
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_RAM:
			t->data = t->host = alloc_mem(bytes, 1);
			if ((t->flag & TENSOR_MASK_CPY) == TENSOR_CPY) {
				if (p)
					memcpy(t->host, p, bytes);
			}
			break;
		case TENSOR_SVM:
			t->context = context;
			t->host = alloc_cl_svm(t->context, CL_MEM_READ_WRITE, bytes, 0);
			t->data = t->host;
			if ((t->flag & TENSOR_MASK_CPY) == TENSOR_CPY) {
				if (p)
					memcpy(t->host, p, bytes);
			}
			break;
		case TENSOR_GPU:
			if ((t->flag & TENSOR_MASK_CPY) == TENSOR_CPY && p) {
				t->data = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, bytes, p, &error->error);
			} else {
				t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &error->error);
			}
			break;
		default:
			fprintf(stderr, "invalid flag tensor\n");
			break;
	}

}


int TensorFill(QUEUE queue, Tensor t, char pattern) {
	return TensorFillOffSet(queue, t, pattern, 0);
}

int TensorFillOffSet(QUEUE queue, Tensor t, char pattern, size_t offset) {
	int erro = 0;
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_RAM:
		case TENSOR_SVM:
			memset(t->host + offset, pattern, t->bytes);
			return erro;

		case TENSOR_GPU:
			erro = clEnqueueFillBuffer(queue, t->data, &pattern, sizeof(char), offset, t->bytes, 0, NULL, NULL);
			PERR(erro, "TensorFillOffSet/clEnqueueWriteBuffer ");
			return erro;
		default:
			erro = TENSOR_INVALID_FLAG_MEM;
			PERR(erro, "TensorFillOffSet: INVALID TENSOR FLAG %d ", t->flag);

	}
}

int TensorFillREAL(QUEUE queue, Tensor t, REAL pattern) {
	return TensorFillREALOffSet(queue, t, pattern, 0);
}

int TensorFillREALOffSet(QUEUE queue, Tensor t, REAL pattern, size_t offset) {
	int erro = 0;
	size_t bytes;
	REAL *mem;
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_RAM:
		case TENSOR_SVM:
			bytes = t->bytes - offset;
			bytes = bytes / sizeof(REAL);
			mem = t->host + offset;
			for (int i = 0; i < bytes; i++) {
				mem[i] = pattern;
			}
			return erro;
		case TENSOR_GPU:
			erro = clEnqueueFillBuffer(queue, t->data, &pattern, sizeof(REAL), offset, t->bytes, 0, NULL, NULL);
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
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_RAM:
		case TENSOR_SVM:
			memcpy(data, t->host + offset, bytes);
			return erro;
		case TENSOR_GPU:
			erro = clEnqueueReadBuffer(queue, t->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueReadBuffer %d 0x%p 0x%p", (int) t->flag, t->data, data);
			return erro;
		default:
			erro = TENSOR_INVALID_FLAG_MEM;
			PERR(erro, "TensorFillOffSet: INVALID TENSOR FLAG %d ", t->flag);

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

int TensorPutValuesMemOffSet(QUEUE queue, Tensor t, void *data, size_t bytes, size_t offset) {
	int erro = 0;
	switch (t->flag & TENSOR_MASK_MEM) {
		case TENSOR_RAM:
		case TENSOR_SVM:
			memcpy(t->host + offset, data, bytes);
			return erro;
		case TENSOR_GPU:
			erro = clEnqueueWriteBuffer(queue, t->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
			PERR(erro, "TensorGetValuesOffSet/clEnqueueReadBuffer %d 0x%p 0x%p", (int) t->flag, t->data, data);
			return erro;
		default:
			erro = TENSOR_INVALID_FLAG_MEM;
			PERR(erro, "TensorFillOffSet: INVALID TENSOR FLAG %d ", t->flag);

	}
}


void printTensor(QUEUE q, Tensor t, FILE *f) {
	union {
		REAL *d;
		int *i;
		unsigned char *c;
	} v;
	v.d = alloc_mem(t->bytes, 1);
	int error = 0;
	for (int l = 0; l < t->w; l++) {
		error = TensorGetValuesOffSet(q, t, v.d, l * t->bytes);
		if (error)break;
		fprintf(f,"{");
		for (int z = 0; z < t->z; z++) {
			fprintf(f,"[");
			for (int i = 0; i < t->x; i++) {
				fprintf(f, "(");
				for (int j = 0; j < t->y; j++){
					switch (t->flag & TENSOR_MASK_TYPE) {
						case TENSOR_REAL:
							fprintf(f, "%.2lf", (double )(v.d[Tensor_Map(t, i, j, z)]));
							break;
						case TENSOR_INT:
							fprintf(f, "%d", v.i[Tensor_Map(t, i, j, z)]);
							break;
						case TENSOR_CHAR:
							fprintf(f, "%02X", (int) v.c[Tensor_Map(t, i, j, z)]);
							break;
						default:
							free_mem(v.d);

					}
					if (j+1 != t->y)
						fprintf(f, ", ");
				}
				fprintf(f, ")");
				if (i+1 != t->x)
					fprintf(f, "\n");
			}
			fprintf(f,"]");
			if (z + 1 != t->z)
				fprintf(f, "\n");
		}
		fprintf(f,"}");
		if(l+1 != t->w)
			fprintf(f,"\n");

	}
	fprintf(f, "\n\n");
	free_mem(v.d);
	if (error) {
		fprintf(stderr, "printTensor: %d\n", error);
//		fprintf(stderr, "printTensor: %d %s", error, getClError(error, buff, EXCEPTION_MAX_MSG_SIZE));
	}

}

int TensorGetNorm(QUEUE queue, Tensor t, REAL *norm) {
	int error = 0;
	if (!norm)return -92;
	if (!t) {
		return -93;
	}
	REAL *v = alloc_mem(t->bytes, t->w);
	error = TensorGetValuesMem(queue, t, v, t->bytes*t->w);
	if (error) {
		fprintf(stderr, "TensorGetNorm/TensorGetValuesMem:");
		free_mem(v);
		return error;
	}
	REAL sum = 0;

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
		case TENSOR_REAL:
			*index = ofset * sizeof(REAL);
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

int TensorCpy(QUEUE queue, Tensor tdst, Tensor tsrc, size_t wsrc) {
	int erro = 0;
	if (tdst->bytes != tsrc->bytes)return TENSOR_INVALID_FLAG_DIM;
	if (wsrc >= tsrc->w)return TENSOR_INVALID_FLAG_DIM;
	void *tmp;
	switch (tdst->flag & TENSOR_MASK_MEM) {
		case TENSOR_RAM:
		case TENSOR_SVM:
			erro = TensorGetValuesMemOffSet(queue, tsrc, tdst->host, tdst->bytes, wsrc * tsrc->bytes);
			PERR(erro, "TensorCpy/clEnqueueReadBuffer ");
			return erro;
		case TENSOR_GPU:
			tmp = alloc_mem(tsrc->bytes, 1);
			erro = TensorGetValuesMemOffSet(queue, tsrc, tmp, tsrc->bytes, wsrc * tsrc->bytes);
			if (erro) {
				free_mem(tmp);
				return erro;
			}
			PERR(erro, "TensorCpy/clEnqueueReadBuffer ");
			erro = TensorPutValuesMemOffSet(queue, tdst, tmp, tdst->bytes, 0);
			free_mem(tmp);
			PERR(erro, "TensorCpy/clEnqueueReadBuffer ");
			return erro;
		default:
			erro = TENSOR_INVALID_FLAG_MEM;
			PERR(erro, "TensorCpy: INVALID TENSOR FLAG %d ", tdst->flag);

	}
	return 0;
}

int TensorRandomize(QUEUE queue, Tensor t, int distribuicao, REAL a, REAL b) {
	REAL *values;
	int length;
	int erro;
	if (!t)return NULL_PARAM;
	values = alloc_mem(t->bytes, t->w);
	length = (t->bytes * t->w) / sizeof(REAL);
	REAL (*X)() = LCG_randD;
	if (distribuicao == LCG_NORMAL) {
		X = LCG_randn;
	}
	for (int i = 0; i < length; ++i) {
		values[i] = X() * a + b;
	}
	erro = TensorPutValuesMem(queue, t, values, t->bytes * t->w);
	free_mem(values);
	return erro;
}


