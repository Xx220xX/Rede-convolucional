//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GABKernel_H
#define GABKernel_H

#include "config.h"
#include<CL/opencl.h>
#include "utils/memory_utils.h"

#define K_VOID_P sizeof(cl_mem)
#define K_INT sizeof(cl_int)
#define K_REAL sizeof(CLREAL)

typedef struct _Kernel {
	void *kernel;
	char *kernel_name;
	int nArgs;
	size_t *l_args;
} *Kernel;

#define QUEUE cl_command_queue

#ifndef EXCEPTION_MAX_MSG_SIZE
#define EXCEPTION_MAX_MSG_SIZE 500
#endif

typedef char char_500[EXCEPTION_MAX_MSG_SIZE] ;

typedef struct Exception {
	cl_int error;
	char_500 msg;
} CNN_ERROR;


Kernel __newKernel(void *pointer_clprogram, CNN_ERROR *error, void *pointer_char_name_function, int n_args, ...);


void __releaseKernel(Kernel *selfp);

int __kernel_run(Kernel, QUEUE, size_t, size_t, ...);

int __kernel_run_recursive(Kernel, QUEUE, size_t, size_t, ...);

void __printKernel(Kernel self);

Kernel
__newKernelHost(void *, CNN_ERROR *, void *, int, ...);


void __releaseKernelHost(Kernel *);


void __printKernelHost(Kernel);

int get_global_id(int);

int set_global_id(int);

#if  (RUN_KERNEL_USING_GPU != 1)
#define __global
#define __kernel
typedef void (*kernel_function_type)(void *, ...);

#define K_ARG
#define new_Kernel(porgram, P_error, kernel_function, n_args, ...) \
		__newKernelHost(kernel_function, P_error,#kernel_function,n_args,## __VA_ARGS__)

#define kernel_run(erro, Kernel, queue, globals, locals, arg0, ...)  \
	   {                                                             \
	   kernel_function_type fc = (kernel_function_type)  Kernel->kernel;         \
	   for (int id=0;id<globals;id++){                       \
				set_global_id(id);                                   \
			fc(arg0,## __VA_ARGS__) ;\
		}                                                       \
		erro  = 0;}


#define kernel_run_recursive(erro, Kernel, queue, globals, maxwords, arg0, ...) \
		kernel_run(erro,Kernel,queue,globals,1,arg0,## __VA_ARGS__,0);

#define releaseKernel __releaseKernelHost
#define printKernel __printKernelHost

#include "float.h"
#define  synchronizeKernel(queue) 0

#else
#define K_ARG &
#define new_Kernel(porgram, P_error, kernel_function, n_args, ...) \
        __newKernel(porgram, P_error,#kernel_function,n_args,## __VA_ARGS__)

#define kernel_run(erro, Kernel, queue, globals, locals, ...)\
        erro= __kernel_run(Kernel,queue,globals,locals,## __VA_ARGS__)

#define kernel_run_recursive(erro, Kernel, queue, globals, max_works, ...)\
        erro = __kernel_run_recursive(Kernel,  queue,  globals,  max_works,## __VA_ARGS__)
#define releaseKernel __releaseKernel
#define printKernel __printKernel
#define  synchronizeKernel(queue) clFinish(queue)
#endif  // RUN_KERNEL_USING_GPU

#endif //GABKernel_H
