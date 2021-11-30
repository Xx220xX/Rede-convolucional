// Created by Xx220xX on 10/05/2020.
#ifndef ATIVATIONSFUNCTIONS_H
#define ATIVATIONSFUNCTIONS_H
#define USEFLOAT 1

#if (USEFLOAT == 1)
#define  REAL float
#define TANH tanh
#define EXP exp
#define SQRT sqrt
#else
#define  REAL double
#define TANH tanh
#define EXP exp
#define SQRT sqrt
#endif
#define Vector __global REAL *

#define kV __kernel void

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

typedef struct {
	int x, y, z;
} Ponto3d;

typedef struct {
	Ponto3d min, max;
} Range;


REAL sigmoid(REAL x) {
	return 1.0 / (1.0 + EXP(-x));
}

REAL difsigmoid(REAL x) {
	REAL tmp = sigmoid(x);
	return tmp * (1.0 - tmp);
}

REAL tanghG(REAL x) {
	return TANH(x);
}

REAL diftanhG(REAL x) {
	REAL tmp = TANH(x);
	return (1.0 - tmp * tmp);
}

REAL relu(REAL x) {
	return x > 0 ? x : 0.0;
}

REAL difrelu(REAL x) {
	return x > 0 ? 1.0 : 0.0;
}

REAL alan(REAL x) {
	if (x > 1) {
		return log10(x)+0.7615941559557649;
	} else if (x < -1) {
		return -log10(-x)-0.7615941559557649;
	}
	return tanghG(x);
}

REAL difalan(REAL x) {
	if (x > 1) {
		return 0.419978 / x;
	} else if (x < 1) {
		return -0.419978 / x;
	}
	return diftanhG(x);
}

REAL func(unsigned int id, REAL x) {
	switch (id) {
		case 0:
			return sigmoid(x);
		case 1:
			return difsigmoid(x);
		case 2:
			return tanghG(x);
		case 3:
			return diftanhG(x);
		case 4:
			return relu(x);
		case 5:
			return difrelu(x);
		case 6:
			return x;
		case 7:
			return 1;
		case 8:
			return alan(x);
		case 9:
			return difalan(x);
		default:
			return 0;
	}
}

#endif