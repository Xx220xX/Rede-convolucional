//
// Created by Xx220xX on 12/05/2020.
//
#include "Kernel.h"
#include "WrapperCL.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int __id_global__kernel = 0;

Kernel __newKernel(void *pointer_clprogram, Exception *error, void *pointer_char_name_function, int n_args, ...) {
	char *f_name = pointer_char_name_function;
	Kernel self = (Kernel) calloc(1, sizeof(struct _Kernel));
	if (error->error)return self;
	self->kernel = clCreateKernel(pointer_clprogram, f_name, &error->error);
	if (error->error) {
		getClErrorWithContext(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE, "new_kernel/%s:", f_name);
		return self;
	}
	if (!self->kernel) {
		error->error = -81;
		snprintf(error->msg, EXCEPTION_MAX_MSG_SIZE, "erro ao criar o kernel : %s\n\t", f_name);
		return self;
	}
	self->l_args = calloc(n_args, sizeof(size_t));
	size_t len_name = strlen(f_name);
	self->nArgs = n_args;
	self->kernel_name = calloc(len_name + 1, sizeof(char));
	memcpy(self->kernel_name, f_name, len_name + 1);

	va_list vaList;
	va_start(vaList, n_args);
	for (int i = 0; i < n_args; ++i) {
		self->l_args[i] = va_arg(vaList, size_t);
	}
	va_end(vaList);
	return self;
}


void __releaseKernel(Kernel *selfp) {
	if (!selfp)return;
	Kernel self = *selfp;
	if (!self)return;
	if (self->kernel)
		clReleaseKernel(self->kernel);
	if (self->l_args)
		free(self->l_args);
	if (self->kernel_name)
		free(self->kernel_name);
	free(self);
	*selfp = NULL;
}

int __kernel_run(Kernel self, cl_command_queue queue, size_t globals, size_t locals, ...) {
	va_list vaList;
	va_start(vaList, locals);
	int error = 0;
	for (int i = 0; i < self->nArgs; ++i) {
		error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		PERR(error, "erro ao colocar argumentos no kernel %s,%d:", self->kernel_name, i);
	}
	va_end(vaList);
	error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
	PERR(error, "erro chamar kernel %s,(%zu,%zu):", self->kernel_name, globals, locals);
	return error;
}

int __kernel_run_recursive(Kernel self, cl_command_queue queue, size_t globals, size_t max_works, ...) {
//	printf("works %zu trabalhos = %zu\n",globals,globals/max_works+globals%max_works);
	va_list vaList;
	int error = 0;
	unsigned int i;
	size_t locals = 1;
	int id = 0;

	va_start(vaList, max_works);
	for (i = 0; i < self->nArgs - 1; i++) {
		error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		PERR(error, "%s: %zu For i = %d arg kernel", self->kernel_name, self->l_args[i], i);
	}
	va_end(vaList);
	error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
	PERR(error, "%s %d: %zu erro ao colocar argumento extra no kernel", self->kernel_name, i, self->l_args[i]);

	if (globals < max_works) {
		locals = globals;
		error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);

	} else {
		size_t resto = globals % max_works;
		globals = (globals / max_works) * max_works;
		locals = max_works;
		error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
		PERR(error, "erro ao rodar kernel %s", self->kernel_name);
		if (resto) {
			id = globals;
			locals = resto;
			globals = resto;

			error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
			PERR(error, "erro ao colocar argumentos no kernel 2 chamada %s", self->kernel_name);
			error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
			PERR(error, "erro ao rodar kernel 2 chamada %s", self->kernel_name);

		}
	}
	PERR(error, "erro ao rodar kernel %s", self->kernel_name);
	return error;
}

void __printKernel(Kernel self) {
	printf("kernel %s: %d args 0x%p[", self->kernel_name, self->nArgs, self->l_args);
	if (self->nArgs > 0) {
		printf("%zu", *self->l_args);
	}
	for (int i = 1; i < self->nArgs; i++) {
		printf(", %zu", self->l_args[i]);
	}
	printf("]\n");
}

Kernel
__newKernelHost(void *pointer_function_kernel, Exception *error, void *pointer_char_name_function, int n_args, ...) {
	char *f_name = pointer_char_name_function;
	Kernel self = (Kernel) calloc(1, sizeof(struct _Kernel));
	if (error->error)return self;
	self->kernel = pointer_function_kernel;
	size_t len_name = strlen(f_name);
	self->nArgs = n_args;
	self->kernel_name = calloc(len_name + 1, sizeof(char));
	memcpy(self->kernel_name, f_name, len_name);
	return self;
}


void __releaseKernelHost(Kernel *selfp) {
	if (!selfp)return;
	Kernel self = *selfp;
	if (!self)return;
	if (self->kernel_name)
		free(self->kernel_name);
	free(self);
	*selfp = NULL;
}


void __printKernelHost(Kernel self) {
	printf("kernel %s: %d ", self->kernel_name, self->nArgs, self->l_args);
	printf("]\n");
}

int get_global_id(int flag) {
	return __id_global__kernel;
}

int set_global_id(int value) {
	__id_global__kernel = value;
	return 0;
}