//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_KERNEL_H
#define GAB_KERNEL_H


#include<CL/cl.h>

#define K_VOID_P sizeof(cl_mem)
#define K_INT sizeof(cl_int)
#define K_DOUBLE sizeof(cl_double)

typedef struct {
	cl_kernel kernel;
	char *kernel_name;
	int nArgs;
	size_t *l_args;
} Kernel;
#ifndef EXCEPTION_MAX_MSG_SIZE
#define EXCEPTION_MAX_MSG_SIZE 500
#endif
typedef struct Exception{
	cl_int error;
	char msg[EXCEPTION_MAX_MSG_SIZE];
} Exception;

Kernel new_Kernel(cl_program pg, Exception *error, const char *f_name, int n_args, ...);

cl_kernel Kernel_putArgs(Kernel *self, int n_args, ...);

int kernel_run(Kernel *self, cl_command_queue queue, size_t globals, size_t locals, ...);

int kernel_run_recursive(Kernel *self, cl_command_queue queue, size_t globals, size_t max_works, ...);

cl_kernel Kernel_get(Kernel *self);

void releaseKernel(Kernel *self);
void printKernel(Kernel *self);
#endif //GAB_KERNEL_H
