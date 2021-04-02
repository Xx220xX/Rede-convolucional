//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_KERNEL_H
#define GAB_KERNEL_H

#include<CL/cl.h>

#define VOID_P sizeof(void *)
#define INT sizeof(cl_int)
#define DOUBLE sizeof(cl_double)

typedef struct {
	cl_kernel kernel;
	int nArgs;
	size_t *l_args;
	int error;
} Kernel;

Kernel new_Kernel(cl_program pg, const char *f_name, int n_args, ...);

cl_kernel Kernel_putArgs(Kernel *self, int n_args, ...);

int kernel_run(Kernel *self, cl_command_queue queue, size_t globals, size_t locals, ...);

int kernel_run_recursive(Kernel *self, cl_command_queue queue, size_t globals, size_t max_works, ...);

cl_kernel Kernel_get(Kernel *self);

void Kernel_release(Kernel *self);

#endif //GAB_KERNEL_H
