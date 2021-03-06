//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_TENSOR_H
#define CNN_GPU_TENSOR_H

#include <stdlib.h>
#include <math.h>
#include"gpu/WrapperCL.h"
#include "utils.h"

size_t max_works = 1;

#ifdef LOG_CNN_TENSOR_MEMORY
#undef LOG_CNN_TENSOR_MEMORY
#define LOG_CNN_TENSOR_MEMORY(format, ...) printf("Tensor memory: ");printf(format,## __VA_ARGS__);printf("\n");
#else
#define LOG_CNN_TENSOR_MEMORY(format, ...)
#endif
typedef struct {
    int x, y, z;
} Ponto3d;

#ifdef TENSOR_TRANSPOSE
#define TensorMap(T, xx, yy, zz) (zz)*((T)->y*(T)->x)+(yy)*(T)->x+(xx)
#else
#define TensorMap(T, xx, yy, zz) (zz)*((T)->y*(T)->x)+(xx)*(T)->y+(yy)
#endif
/***
 * Tensor armazena uma matriz 3D juntamente com os parametros dela
 */
typedef struct {
    cl_mem data;
    unsigned int bytes, x, y, z;
} *Tensor, typetensor, *TensorChar, typetensorchar;
typedef struct {
    int x, y, z;
    double *data;
} *TensorC, typeTensorC;

void fillTensor(Tensor t, cl_context context, size_t bytes, GPU_ERROR *error) {
    t->data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &error->error);
    if (!t->data) {
        error->error = -1;
        snprintf(error->msg, 255, "A memoria retornada foi NULL\n");
    }
    if (error->error) {
        snprintf(error->msg, 255, "nao foi possivel allocar memoria vram\n");

    }
    LOG_CNN_TENSOR_MEMORY("aloc (0x%X,0x%X)", t, t->data)
}

Tensor newTensor(cl_context context, unsigned int x, unsigned int y, unsigned int z, GPU_ERROR *error) {
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

void dividirVetor(double *v, cl_mem m, size_t len, double value, Kernel funcNorm, cl_command_queue queue) {
    int error = 0, id = 0;
    size_t global, local, resto;
    clEnqueueWriteBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
    call_kernel(len,
                Kernel_putArgs(&funcNorm, 3, &m, &value, &id);
                        error = clEnqueueNDRangeKernel(queue, funcNorm.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel div")
    );
    clFinish(queue);
    clEnqueueReadBuffer(queue, m, CL_TRUE, 0, len * sizeof(double), v, 0, NULL, NULL);
}

void dividirVetorInt(unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, double value, Kernel funcNorm, cl_command_queue queue) {
    int error = 0, id = 0;
    size_t global, local, resto;
    clEnqueueWriteBuffer(queue, mi, CL_TRUE, 0, len * sizeof(unsigned char), src, 0, NULL, NULL);
    call_kernel(len,
                Kernel_putArgs(&funcNorm, 3, &mi, &mout, &value, &id);
                        error = clEnqueueNDRangeKernel(queue, funcNorm.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel divInt")
    );
    clFinish(queue);
    clEnqueueReadBuffer(queue, mout, CL_TRUE, 0, len * sizeof(double), dst, 0, NULL, NULL);

}

void int2doubleVector(WrapperCL *cl, unsigned char *src, double *dst, cl_mem mi, cl_mem mout, size_t len, int nop, Kernel func, cl_command_queue queue) {
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

    int error = 0, id = 0;
    size_t global=1, local=1, resto;
    clEnqueueWriteBuffer(queue, mi, CL_TRUE, 0, len * sizeof(unsigned char), src, 0, NULL, NULL);
    /*
     * Kernel_putArgs(&printI, 2, &mi, &len);
    clEnqueueNDRangeKernel(queue, printI.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
   */
    call_kernel(len,
                Kernel_putArgs(&func, 3, &mi, &mout, &nop, &id);
                        error = clEnqueueNDRangeKernel(queue, func.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel divInt")
    );
    clFinish(queue);
   /* int g = 10;
    global = 1;local = 1;
    Kernel_putArgs(&printD, 3, &mout, &len,&g);
    clEnqueueNDRangeKernel(queue, printD.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clReleaseProgram(pg);
    Kernel_release(&printI);
    Kernel_release(&printD);
    */
    clEnqueueReadBuffer(queue, mout, CL_TRUE, 0, len *10* sizeof(double), dst, 0, NULL, NULL);
}

#endif //CNN_GPU_TENSOR_H
