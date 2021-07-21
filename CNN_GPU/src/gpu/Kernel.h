//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GABKernel _H
#define GABKernel _H

//#define DISABLE_KERNELS_INSIDE_DRIVE

#include<CL/cl.h>

#define K_VOID_P sizeof(cl_mem)
#define K_INT sizeof(cl_int)
#define K_DOUBLE sizeof(cl_double)

typedef struct _Kernel *Kernel;


#ifndef EXCEPTION_MAX_MSG_SIZE
#define EXCEPTION_MAX_MSG_SIZE 500
#endif
typedef struct Exception {
	cl_int error;
	char msg[EXCEPTION_MAX_MSG_SIZE];
} Exception;

Kernel __newKernel(void *, Exception *error, void *f_name, int n_args, ...);


void releaseKernel(Kernel *self);


void printKernel(Kernel self);

#ifndef DISABLE_KERNELS_INSIDE_DRIVE
/// Usado nos argumentos passados para rodar um kernel
#define K_ARG &
#define new_Kernel(porgram, P_error, kernel_function, n_args, ...) \
        __newKernel(porgram, P_error,#kernel_function,n_args,## __VA_ARGS__)

#define kernel_run(erro, Kernel, queue, globals, locals, ...)\
        erro= __kernel_run(Kernel,queue,globals,locals,## __VA_ARGS__)

#define kernel_run_recursive(erro, Kernel, queue, globals, locals, ...)\
        erro = __kernel_run_recursive(Kernel,queue,globals,locals,## __VA_ARGS__)
#define KernelSincronise clFinish
int __kernel_run(Kernel self, cl_command_queue queue, size_t globals, size_t locals, ...);

int __kernel_run_recursive(Kernel self, cl_command_queue queue, size_t globals, size_t max_works, ...);

#else
'	#define K_ARG
#define newKernel(porgram, P_error, kernel_function, n_args, ...) \
		__newKernel(porgram, P_error,#kernel_function,n_args,## __VA_ARGS__)

#define kernel_run(Kernel, queue, globals, locals, ...) { \
		for (int id=0;id<globals;id++){                       \
				set_global_id(id);                                              \
			(kernel)->,queue,globals,locals,## __VA_ARGS__) ;
	}\
	}
#define kernel_run_recursive(Kernel, queue, globals,maxwords, ...) \
		kernel_run(Kernel,queue,globals,1,## __VA_ARGS__);

	int get_global_id(int flag);
'int set_global_id(int value);
#endif  // DISABLE_KERNELS_INSIDE_DRIVE


#endif //GABKernel _H
