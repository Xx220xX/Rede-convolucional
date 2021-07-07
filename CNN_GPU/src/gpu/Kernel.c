//
// Created by Xx220xX on 12/05/2020.
//

#include "Kernel.h"
#include "WrapperCL.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

Kernel new_Kernel(cl_program pg, GPU_ERROR *error, const char *f_name, int n_args, ...) {
	Kernel self = {0};
	if (error->error)return self;
	self.kernel = clCreateKernel(pg, f_name, &error->error);
	if(error->error)
		getClError(error->error, error->msg + sprintf(error->msg, "erro ao criar o kernel : %s\n\t", f_name));
	self.l_args = calloc(n_args, sizeof(size_t));
	size_t len_name = strlen(f_name);
	self.nArgs = n_args;
	self.kernel_name = calloc(len_name + 1, sizeof(char));
	memcpy(self.kernel_name, f_name, len_name + 1);

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
		PER(error, "erro ao colocar argumentos no kernel", self->kernel_name);
	}
	va_end(vaList);
	return self->kernel;
}

cl_kernel Kernel_get(Kernel *self) {
	return self->kernel;
}

void releaseKernel(Kernel *self) {
	if (self->kernel)
		clReleaseKernel(self->kernel);
	if (self->l_args)
		free(self->l_args);
	if (self->kernel_name)
		free(self->kernel_name);
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
		PERR(error, "erro ao colocar argumentos no kernel", self->kernel_name);
	}
	va_end(vaList);
	error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
	return error;
}

int kernel_run_recursive(Kernel *self, cl_command_queue queue, size_t globals, size_t max_works, ...) {
	va_list vaList;
	int error = 0;
	int i;
	size_t locals = 1;
	int id = 0;

	va_start(vaList, max_works);
	for (i = 0; i < self->nArgs - 1; ++i) {
		error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		PERR(error, "erro ao colocar argumentos no kernel ", self->kernel_name);
	}
	va_end(vaList);
	error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
	PERR(error, "erro ao colocar argumentos no kernel", self->kernel_name);

	if (globals < max_works) {
		locals = globals;
		error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
	} else {
		size_t resto = globals % max_works;
		globals = (globals / max_works) * max_works;
		locals = max_works;
		error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);

		PERR(error, "erro ao rodar kernel", self->kernel_name);
		if (resto) {
			id = globals;
			locals = resto;
			globals = resto;

			error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
			PERR(error, "erro ao colocar argumentos no kernel 2 chamada", self->kernel_name);

			error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
		}
	}
	PERR(error, "erro ao rodar kernel _", self->kernel_name);
	return error;
}

