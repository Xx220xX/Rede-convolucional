//
// Created by Henrique on 23-Jul-21.
//

#ifndef CNN_GPU_MEMORY_UTILS_H
#define CNN_GPU_MEMORY_UTILS_H

#include <CL/opencl.h>
#include <stdlib.h>

void *__alloc_cl_svm__(cl_context context,
                       cl_svm_mem_flags flags,
                       size_t size,
                       cl_uint alignment);

void __free_cl_svm(cl_context context, void *svm_pointer);

void *__alloc_mem_(size_t elements, size_t size_element);

void __free_mem_(void *mem_pointer);

void *__realloc_mem_(void *mem_pointer, size_t new_size);

char *printBytes(size_t bytes, char buff[250]);

void releaseMemWatcher();

void printMemStatus();

#define  MEMORY_WATCHER
#ifdef  MEMORY_WATCHER
#define alloc_cl_svm __alloc_cl_svm__
#define free_cl_svm __free_cl_svm

#define alloc_mem __alloc_mem_
#define free_mem __free_mem_
#define realloc_mem __realloc_mem_

int __main__(int arg, char **args);

#ifndef MAINCREATE
#define MAINCREATE

int main(int arg, char **args);

#endif //MAINCREATE
#define main __main__


#else
#define alloc_cl_svm clSVMalloc
#define free_cl_svm clSVMFree
#define alloc_mem calloc
#define free_mem free
#define realloc_mem realloc
#endif //  MEMORY_WATCHER
#endif //CNN_GPU_MEMORY_UTILS_H
