//
// Created by hslhe on 14/11/2021.
//

#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "tensor/exc.h"
#include "stdlib.h"

void Ecx_popstack(Ecx self);

void Ecx_addstack(Ecx self, const char *stack);

void Ecx_release(Ecx *self_p);

void Ecx_print(Ecx self);

int Ecx_setError(Ecx self, int error, char *msg, ...);

void vEcx_pushMsg(Ecx self, const char *format, va_list v);

void Ecx_pushMsg(Ecx self, const char *format, ...) {
	va_list v;
	va_start(v, format);
	vEcx_pushMsg(self, format, v);
	va_end(v);
}

void vEcx_pushMsg(Ecx self, const char *format, va_list v) {
	int len = vsnprintf(NULL, 0, format, v) + 1;
	if (self->msg) {
		free(self->msg);
	}
	self->msg = malloc(len);
	vsnprintf(self->msg, len, format, v);
}

Ecx Ecx_new(int stack_len) {
	Ecx self = calloc(1, sizeof(Ecx_t));
	self->len = stack_len;
	self->index = -1;
	self->perro = &self->error;
	if (stack_len > 0) {
		self->stack = calloc(stack_len, sizeof(char *));
	}
	self->popstack = Ecx_popstack;
	self->addstack = Ecx_addstack;
	self->release = Ecx_release;
	self->print = Ecx_print;
	self->setError = Ecx_setError;
	return self;
}

int Ecx_setError(Ecx self, int error, char *format,...) {
	if (self->error) {
		return self->error;
	}
	while (self->block);
	self->block = 1;
	self->error = error;
	va_list v;
	va_start(v, format);
	vEcx_pushMsg(self, format,v);
	va_end(v);
	self->block = 0;
	return self->error;
}

void Ecx_addstack(Ecx self, const char *stack) {
	if (self->error) {
		return;
	}
	if (self->len < 0) {
		return;
	}
	while (self->block);
	self->block = 1;
	self->index++;
	if (self->index >= self->len) {
		self->len = self->index + 1;
		self->stack = realloc(self->stack, self->len * sizeof(char *));
	}
	unsigned long long l = strlen(stack);
	self->stack[self->index] = calloc(l + 1, sizeof(char));
	memcpy(self->stack[self->index], stack, l);
	self->block = 0;

}

void Ecx_popstack(Ecx self) {
	if (self->error) {
		return;
	}
	if (self->len <= 0) {
		return;
	}
	if (self->index < 0) {
		return;
	}
	while (self->block);
	self->block = 1;
	free(self->stack[self->index]);
	self->index--;
	self->block = 0;
}

void Ecx_release(Ecx *self_p) {
	if (!self_p) {
		return;
	}
	if (!*self_p) {
		return;
	}
	for (int i = 0; i <= (*self_p)->index; ++i) {
		if ((*self_p)->stack[i]) {
			free((*self_p)->stack[i]);
		}
	}
	if ((*self_p)->stack) {
		free((*self_p)->stack);
	}
	if ((*self_p)->msg) {
		free((*self_p)->msg);
	}
	free((*self_p));
}

void Ecx_print(Ecx self) {
	while (self->block);
	self->block = 1;
	if (self->error) {
		printf("Erro: %d: %s\n", self->error, self->msg);
	}
	for (int i = 0; i <= self->index; ++i) {
		printf("%d:%s\n", i + 1, self->stack[i]);
	}
	self->block = 0;
}
