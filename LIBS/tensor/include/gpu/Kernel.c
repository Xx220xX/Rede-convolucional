//
// Created by Xx220xX on 12/05/2020.
//
#include "gpu/Kernel.h"
#include "gpu/Gpu.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define check_error(error, end, format, ...)if(error){char *msg = Gpu_errormsg(error);fflush(stdout); \
fprintf(stderr,"Error %d  in file %s at line %d.\n%s\n",error,__FILE__,__LINE__,msg);           \
fprintf(stderr,format,## __VA_ARGS__);free_mem(msg);goto end;}

#define PAD " "
#define apendstr(str, len, format, ...) { \
         size_t sz = snprintf(NULL,0,format,##__VA_ARGS__); \
         if(!str)                         \
         str = alloc_mem(1,sz+1);    \
         else                                 \
         str = realloc(str,len+sz+1);                              \
         char *tmp = str+len;               \
         len = len+sz;\
         sprintf(tmp,format,##__VA_ARGS__) ;                           \
}

void Kernel_release(Kernel *self) {
	if (!self)return;
	if (!(*self))return;
	if ((*self)->kernel)
		clReleaseKernel((*self)->kernel);
	if ((*self)->l_args)
		free_mem((*self)->l_args);
	if ((*self)->name)
		free_mem((*self)->name);
	free_mem(*self);
}

int Kernel_run(Kernel self, cl_command_queue queue, size_t globals, size_t locals, ...) {
	va_list vaList;
	va_start(vaList, locals);
	int error = 0;
	for (int i = 0; i < self->nArgs; ++i) {
		error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		check_error(error, end, "erro ao colocar argumentos no kernel %s,%d:", self->name, i);
	}
	va_end(vaList);
	error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
	check_error(error, end, "erro chamar kernel %s,(%zu,%zu):", self->name, globals, locals);
	end:
	return error;
}

int Kernel_runRecursive(Kernel self, cl_command_queue queue, size_t globals, size_t max_works, ...) {
	if (self->error)return self->error;
	va_list vaList;
	int error = 0;
	unsigned int i;
	size_t locals = 1;
	int id = 0;

	va_start(vaList, max_works);
	for (i = 0; i < self->nArgs - 1; i++) {
		self->error = clSetKernelArg(self->kernel, i, self->l_args[i], va_arg(vaList, void *));
		check_error(error, end, "kernel %s: %zu For i = %d arg kernel", self->name, self->l_args[i], i);
	}
	va_end(vaList);
	self->error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
	check_error(self->error, end, "Kernel %s %d: %zu erro ao colocar argumento extra no kernel", self->name, i, self->l_args[i]);

	if (globals < max_works) {
		locals = globals;
		self->error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
		check_error(self->error, end, "Kernel %s", self->name);
	} else {
		size_t resto = globals % max_works;
		globals = (globals / max_works) * max_works;
		locals = max_works;
		self->error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
		check_error(self->error, end, "erro ao rodar kernel %s", self->name);
		if (resto) {
			id = globals;
			locals = resto;
			globals = resto;

			self->error = clSetKernelArg(self->kernel, i, self->l_args[i], &id);
			check_error(self->error, end, "erro ao colocar argumentos no kernel 2 chamada %s", self->name);
			self->error = clEnqueueNDRangeKernel(queue, self->kernel, 1, NULL, &globals, &locals, 0, NULL, NULL);
			check_error(self->error, end, "erro ao rodar kernel 2 chamada %s", self->name);

		}
	}

	end:
	return error;
}

char *Kernel_json(Kernel self) {
	char *json;
	int len = 0;
	apendstr(json, len, "{\n"
			PAD"\"name\":\"%s\",\n"
			PAD"nArgs:\"%d\",\n"
			PAD"\"l_args\":[", self->name, self->nArgs
	);

	for (int i = 0; i < self->nArgs; ++i) {
		if (i == 0) {
			apendstr(json, len, "%zu", self->l_args[i]);
		} else {
			apendstr(json, len, ", %zu", self->l_args[i]);
		}
	}
	apendstr(json, len, "]\n}");

	return json;
}

Kernel Kernel_new(cl_program clProgram, char *funcname, int nargs, ...) {
	Kernel self = (Kernel) alloc_mem(1, sizeof(Kernel_t));
	size_t len_name = strlen(funcname);
	self->name = alloc_mem(len_name + 1, 1);
	memcpy(self->name, funcname, len_name);
	self->kernel = clCreateKernel(clProgram, funcname, &self->error);
	check_error(self->error, metods, "Kernel: %s\n", funcname);

	self->l_args = alloc_mem(nargs, sizeof(size_t));
	self->nArgs = nargs;
	va_list vaList;
	va_start(vaList, nargs);
	for (int i = 0; i < nargs; ++i) {
		self->l_args[i] = va_arg(vaList, size_t);
	}
	va_end(vaList);
	metods:
	self->release = (void (*)(void *)) Kernel_release;
	self->run = (void (*)(void *, cl_command_queue, size_t, size_t, ...)) Kernel_run;
	self->runRecursive = (void (*)(void *, cl_command_queue, size_t, size_t, ...)) Kernel_runRecursive;
	self->json = (char *(*)(void *)) Kernel_json;
	return self;
}
