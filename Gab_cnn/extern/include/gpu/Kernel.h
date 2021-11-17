//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GABKernel_H
#define GABKernel_H


#include<CL/opencl.h>
#include "config.h"

#define KP sizeof(cl_mem)
#define KI sizeof(cl_int)
#define KR sizeof(CLREAL)

typedef struct Kernel_t {
	void *kernel;
	char *name;
	size_t *l_args;
	int nArgs;
	int error;

	char *(*json)(struct  Kernel_t*self);

	void (*release)(struct  Kernel_t**self_p);

	int (*run)(struct  Kernel_t*self_p, cl_command_queue queue, size_t globals, size_t locals, ...);

	int (*runRecursive)(struct  Kernel_t*self, cl_command_queue queue, size_t globals, size_t max_works, ...);
} *Kernel, Kernel_t;


extern Kernel Kernel_new(cl_program clProgram, char *funcname, int nargs, ...);

extern Kernel Kernel_news(cl_program clProgram, char *funcname, const char *p);


#define __kernel
#define __global
#define  Vector __global REAL *
#define KV __kernel void
#define KTensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))

#define KTensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))

#define KTensorRemap4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\
_y_ = total%ty      ;                                        \
_x_ = (total - _y_)%(ty*tx)/ty ;                             \
_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \
_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);


#define KTensorRemap(total, _x_, _y_, _z_, tx, ty)\
_y_ = total % ty;\
_x_ = ((total - _y_) % (ty * tx)) / ty;\
_z_ = (k - _x_ * ty - _y_) / (tx * ty);

#define KTensorRemap2D(total, x, y, ty)\
y = total % ty;\
x = total/ ty;


#define  UTILS_MACRO_KERNEL \
KREAL "#define  Vector __global REAL *\n"\
"#define KV __kernel void\n"\
"#define KTensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))\n"\
"\n"\
"#define KTensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))\n"\
"\n"\
"#define KTensorRemap4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\\\n"\
"_y_ = total%ty      ;                                        \\\n"\
"_x_ = (total - _y_)%(ty*tx)/ty ;                             \\\n"\
"_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \\\n"\
"_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);\n"\
"\n"\
"\n"\
"#define KTensorRemap(total, _x_, _y_, _z_, tx, ty)\\\n"\
"_y_ = total % ty;\\\n"\
"_x_ = ((total - _y_) % (ty * tx)) / ty;\\\n"\
"_z_ = (k - _x_ * ty - _y_) / (tx * ty);\n"\
"\n"\
"#define KTensorRemap2D(total, x, y, ty)\\\n"\
"y = total % ty;\\\n"\
"x = total/ ty;\n\n"

#define PAD " "
#define apendstr(str, len, format, ...) { \
         size_t sz = snprintf(NULL,0,format,##__VA_ARGS__); \
         if(!str)                         \
         str = alloc_mem(1,sz+1);    \
         else                                 \
         str = realloc(str,len+sz+1);                              \
         char *local_tmp = str+len;               \
         len = len+sz;\
         sprintf(local_tmp,format,##__VA_ARGS__) ;                           \
}
#endif //GABKernel_H
