//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GABKernel_H
#define GABKernel_H

#include<CL/opencl.h>
#include "config.h"

typedef struct Kernel_t {
	void *kernel;
	char *name;
	size_t *l_args;
	int nArgs;
	int error;

	char *(*json)(struct Kernel_t *self);

	void (*release)(struct Kernel_t **self_p);

	int (*run)(struct Kernel_t *self_p, cl_command_queue queue, size_t globals, size_t locals, ...);

	int (*runRecursive)(struct Kernel_t *self, cl_command_queue queue, size_t globals, size_t max_works, ...);
} *Kernel, Kernel_t;

extern Kernel Kernel_new(cl_program clProgram, char *funcname, int nargs, ...);

extern Kernel Kernel_news(cl_program clProgram, char *funcname, const char *p);


#endif //GABKernel_H
