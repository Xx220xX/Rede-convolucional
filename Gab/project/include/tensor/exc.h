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

	int (*setError)(struct Ecx_t *self, int error, char *format, ...);

	void (*addstack)(struct Ecx_t *self, const char *stack);//__attribute__((deprecated));

	void (*popstack)(struct Ecx_t *self);//__attribute__((deprecated));

	void (*release)(struct Ecx_t **self_p);

	void (*pushMsg)(struct Ecx_t *self, const char *format, ...);

	void (*print)(struct Ecx_t *self);
} *Ecx, Ecx_t;

Ecx Ecx_new(int stack_len);

#define ECX_REGISTRE_FUNCTION_IF_ERROR(ecx) if(ecx->error){(ecx)->addstack(ecx,__FUNCTION__);}
#define ECX_RETURN_IF_ERROR(ecx, value_return)if(!ecx->error){return value_return;}
#define ECX_TRY(ecx,cond,goto_point,error,msg,...)if(cond){ ecx->setError(ecx,error,__FILE__":%d" " " msg,__LINE__,##__VA_ARGS__);goto goto_point;}
#define ECX_IF_FAILED(ecx,goto_failed)if(ecx->error){goto goto_failed;}


#define ECXCHECKAFTER(ecx, goto_point, func, arg, ...)func(arg,##__VA_ARGS__);if(ecx->error){fprintf(stderr,"ecx %d in %s %s:%d\n",ecx->error,#func,__FILE__,__LINE__);goto goto_point;}


#endif //TENSOR_EXC_H
