//
// Created by hslhe on 14/11/2021.
//

#ifndef TENSOR_EXC_H
#define TENSOR_EXC_H

#include <stdatomic.h>

typedef struct Ecx_t {
	int error;
	int *perro;
	char **stack;
	char *msg;
	int len;
	int index;

	atomic_int block;

	int (*setError)(struct Ecx_t *self, int error, char *format,...);

	void (*addstack)(struct Ecx_t *self, const char *stack);

	void (*popstack)(struct Ecx_t *self);

	void (*release)(struct Ecx_t **self_p);

	void (*pushMsg)(struct Ecx_t *self, const char *format, ...);

	void (*print)(struct Ecx_t *self);
} *Ecx, Ecx_t;

Ecx Ecx_new(int stack_len);

#define ECXCHECKAFTER(ecx, goto_point, func, arg, ...)func(arg,##__VA_ARGS__);if(ecx->error){fprintf(stderr,"erro %d in %s %s:%d\n",ecx->error,#func,__FILE__,__LINE__);goto goto_point;}
#define ECXRUN(ecx, func, arg, ...)ecx->addstack(ecx,#func);func( arg,##__VA_ARGS__ );ECXPOP(ecx)
#define ECXPUSH(ecx)(ecx)->addstack(ecx,__FUNCTION__)
#define ECXPOP(ecx)(ecx)->popstack(ecx)

#endif //TENSOR_EXC_H
