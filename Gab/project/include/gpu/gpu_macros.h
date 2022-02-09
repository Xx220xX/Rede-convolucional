//
// Created by Henrique on 01/01/2022.
//

#ifndef GAB_GPU_MACROS_H
#define GAB_GPU_MACROS_H
/// memória de  escrita
#define Vw __global REAL *
/// memória de leitura
#define Vr __global REAL *
/// memória de leitura e ecrita
#define Vrw __global REAL *

#define kV __kernel void

#define kMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))

#define kMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))

#define kRep4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\
_y_ = total%ty      ;                                        \
_x_ = (total - _y_)%(ty*tx)/ty ;                             \
_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \
_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);


#define kRap(total, _x_, _y_, _z_, tx, ty)\
_y_ = total % ty;\
_x_ = ((total - _y_) % (ty * tx)) / ty;\
_z_ = (total - _x_ * ty - _y_) / (tx * ty);

#define KRap2D(total, x, y, ty)\
y = total % ty;\
x = total/ ty;

#endif //GAB_GPU_MACROS_H
