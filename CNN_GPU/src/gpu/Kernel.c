//
// Created by Xx220xX on 12/05/2020.
//

#include "Kernel.h"
#include "WrapperCL.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

Kernel new_Kernel(cl_program pg, const char *f_name, int n_args, ...) {
	int error = 0;
	Kernel self = {0};
	self.kernel = clCreateKernel(pg, f_name, &error);
	PER(error, "erro ao criar o kernel");
	self.nArgs = n_args;
	self.l_args = calloc(n_args, sizeof(size_t));

	va_list vaList;
	va_start(vaList, n_args);
	for (int i = 0; i < n_args; ++i) {
		self.l_args[i] = va_arg(vaList, int);

	}
	va_end(vaList);
	return self;
}

cl_kernel Kernel_putArgs(Kernel *self, int n_args, ...) {
	va_list vaList;
	va_start(vaList, n_args);
	int error = 0;
	for (int i = 0; i < self->nArgs; ++i) {
		error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		PER(error, "erro ao colocar argumentos no kernel");
	}
	va_end(vaList);
	return self->kernel;
}

cl_kernel Kernel_get(Kernel *self) {
	return self->kernel;
}

void Kernel_release(Kernel *self) {
	if (self->kernel)
		clReleaseKernel(self->kernel);
	if (self->l_args)
		free(self->l_args);
	self->nArgs = 0;
	self->l_args = NULL;
	self->kernel = NULL;
}

int kernel_run(Kernel *self, cl_command_queue queue, size_t globals, size_t locals, ...) {
	va_list vaList;
	va_start(vaList, locals);
	int error = 0;
	for (int i = 0; i < self->nArgs; ++i) {
		error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		PERR(error, "erro ao colocar argumentos no kernel");
	}
	va_end(vaList);
	error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
	return error;
}
int kernel_run_recursive(Kernel *self, cl_command_queue queue, size_t globals, size_t max_works, ...) {
	va_list vaList;
	va_start(vaList, max_works);
	int error = 0;
	int i;
	for (i = 0; i < self->nArgs; ++i) {
		error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		PERR(error, "erro ao colocar argumentos no kernel");
	}
	va_end(vaList);
	size_t locals = 1;
	int id = 0;
	error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
	PERR(error, "erro ao colocar argumentos no kernel");
	if (globals < max_works) {
		locals = globals;
		error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
	} else {
		size_t resto = globals % max_works;
		globals = globals - resto;
		locals = globals;
		error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
		PERR(error, "erro ao rodar kernel");
		if (resto) {
			id = globals;
			locals = globals;
			error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
			PERR(error, "erro ao colocar argumentos no kernel");
			globals = resto;
			error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
		}
	}
	PERR(error, "erro ao rodar kernel _");
	return error;
}

