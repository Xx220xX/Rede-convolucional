//
// Created by hslhe on 14/11/2021.
//

#ifndef TENSOR_EXC_H
#define TENSOR_EXC_H

#include <stdatomic.h>

typedef struct Ecx_t {
	int error;
	char **stack;
	int len;
	int index;

	atomic_int block;

	void (*addstack)(struct Ecx_t *self, char *stack);

	void (*popstack)(struct Ecx_t *self);

	void (*release)(struct Ecx_t **self_p);

	void (*print)(struct Ecx_t *self);
} *Ecx, Ecx_t;

Ecx Ecx_new(int stack_len);

#endif //TENSOR_EXC_H
