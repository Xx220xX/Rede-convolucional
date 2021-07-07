//
// Created by petel on 22/05/2021.
//

#include "Tensor.h"
GPU_ERROR defaultError;

void fillTensor(Tensor t, cl_context context, size_t bytes, GPU_ERROR *error) {
    t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &error->error);
	if (!t->data) {
	    snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
    }
    if (error->error) {
        snprintf(error->msg, 255, "nao foi possivel allocar memoria vram\n");
    }
    LOG_CNN_TENSOR_MEMORY("aloc (0x%X,0x%X)", t, t->data)
}


int TensorPutValues(cl_command_queue queue, Tensor t, void *data) {
    return clEnqueueWriteBuffer(queue, t->data, CL_TRUE, 0, t->bytes, data, 0, NULL, NULL);
}int TensorPutValuesOffSet(cl_command_queue queue, Tensor t, void *data,UINT ofset) {
    return clEnqueueWriteBuffer(queue, t->data, CL_TRUE,ofset, t->bytes, data, 0, NULL, NULL);
}


void TensorGetValues(cl_command_queue queue, Tensor t, void *data) {
    clEnqueueReadBuffer(queue, t->data, CL_TRUE, 0, t->bytes, data, 0, NULL, NULL);
}
void TensorGetValuesOffset(cl_command_queue queue, Tensor t,int offset, void *data) {
    clEnqueueReadBuffer(queue, t->data, CL_TRUE, offset, t->bytes, data, 0, NULL, NULL);
}


Tensor newTensor(cl_context context, unsigned int x, unsigned int y, unsigned int z, GPU_ERROR *error) {
    if (!error)error = &defaultError;
    if (error->error)return NULL;
    if (x <= 0 | y <= 0 | z <= 0) {
        error->error = -1;
    }
    Tensor t = (Tensor) calloc(1, sizeof(typetensor));
    t->bytes = x * y * z * sizeof(double);
    t->x = x;
    t->y = y;
    t->z = z;
    fillTensor(t, context, t->bytes, error);
    return t;
}


Tensor newTensor4D(cl_context context, unsigned int x, unsigned int y, unsigned int z, unsigned int l, GPU_ERROR *error) {
    if (!error)error = &defaultError;
    if (error->error)return NULL;
    Tensor t = (Tensor) calloc(1, sizeof(typetensor));
    t->bytes = x * y * z * sizeof(double);
    t->x = x;
    t->y = y;
    t->z = z;
    fillTensor(t, context, t->bytes * l, error);
    return t;
}


TensorChar newTensorChar(cl_context context, unsigned int x, unsigned int y, unsigned int z, GPU_ERROR *error) {
    if (!error)error = &defaultError;
    if (error->error)return NULL;
    TensorChar t = (Tensor) calloc(1, sizeof(typetensorchar));
    t->bytes = x * y * z * sizeof(char);
    t->x = x;
    t->y = y;
    t->z = z;
    fillTensor(t, context, t->bytes, error);
    return t;
}


void releaseTensor(Tensor *t) {
    if (*t) {
        LOG_CNN_TENSOR_MEMORY("free (0x%X,0x%X)", *t, (*t)->data)
        if ((*t)->data)
            clReleaseMemObject((*t)->data);
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


void
dividirVetor(double *v, cl_mem m, size_t len, double value, Kernel funcNorm, size_t max_works, cl_command_queue queue) {
    int error = 0, id = 0;
    size_t global, local, resto;
    clEnqueueWriteBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
    kernel_run_recursive(&funcNorm, queue, len, max_works, &m, &value);
    clFinish(queue);
    clEnqueueReadBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
}



void dividirVetorInt(unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, double value, Kernel funcNorm,
                     size_t max_works, cl_command_queue queue) {
    clEnqueueWriteBuffer(queue, mi, CL_TRUE, 0, len * sizeof(unsigned char), src, 0, NULL, NULL);
    kernel_run_recursive(&funcNorm, queue, len, max_works, &mi, &mout, &value);
    clFinish(queue);
    clEnqueueReadBuffer(queue, mout, CL_TRUE, 0, len * sizeof(double), dst, 0, NULL, NULL);
}



void int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, int nop,
                      Kernel func, cl_command_queue queue) {
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

