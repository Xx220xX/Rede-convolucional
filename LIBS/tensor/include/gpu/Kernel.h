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

	char *(*json)(void *self);

	void (*release)(void *self_p);

	void (*run)(void *self_p, cl_command_queue queue, size_t globals, size_t locals, ...);

	void (*runRecursive)(void *self, cl_command_queue queue, size_t globals, size_t max_works, ...);
} *Kernel, Kernel_t;


extern Kernel Kernel_new(cl_program clProgram, char *funcname, int nargs, ...);


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
"x = total/ ty;\n\n"\

#endif //GABKernel_H
