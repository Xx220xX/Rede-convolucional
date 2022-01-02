#include "kernels.h"
//utils.h
// Created by Xx220xX on 10/05/2020.
#ifndef ATIVATIONSFUNCTIONS_H
#define ATIVATIONSFUNCTIONS_H
#define USEFLOAT 1

#if (USEFLOAT == 1)
#define    REAL float
#define    EXP exp
#define    SQRT sqrt
#define    REALMAX FLT_MAX
#define    REALMIN FLT_MIN
#else
#define	REALMAX DBL_MAX
#define	REALMIN DBL_MIN
#define	REAL double
#define	EXP exp
#define	SQRT sqrt
#endif
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
_z_ = (k - _x_ * ty - _y_) / (tx * ty);

#define KRap2D(total, x, y, ty)\
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
	return tanh(x);
}

REAL diftanhG(REAL x) {
	REAL tmp = tanh(x);
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
		return log10(x) + 0.7615941559557649;
	} else if (x < -1) {
		return -log10(-x) - 0.7615941559557649;
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

//bathnorm.h
kV BatchNormMedia(Vr a, Vw u, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	int x, y;
	REAL m = 0;
	for (x = 0; x < ax; x++) {
		for (y = 0; y < ay; y++) {
			m += a[kMap(x, y, z, ax, ay)];
		}
	}
	u[z] = m / (REAL) (ax * ay);
}

kV BatchNormInvDesv(Vr a, Vr u, Vr o, REAL episolon, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	REAL sum = 0;
	REAL tmp;
	for (int x = 0; x < ax; x++) {
		for (int y = 0; y < ay; y++) {
			tmp = (a[kMap(x, y, z, ax, ay)] - u[z]);
			sum += tmp * tmp;
		}
	}
	sum = sum / (ax * ay);
	o[z] = 1.0 / sqrt(sum + episolon);
}

kV BatchNormNormaliza(Vw s, Vrw v, Vr a, Vr u, Vr o, Vr Y, Vr B, int ax, int ay, int id_0) {
	int x, y, z;
	int k = get_global_id(0) + id_0;
	kRap(k, x, y, z, ax, ay)
	v[k] = (a[k] - u[z]) * o[z];
	s[k] = v[k] * Y[z] + B[z];
}

kV BatchNormaCalcDnorm(Vw dv, Vr ds,Vr Y, int ax, int ay, int id_0) {
	int x, y, z;
	int k = get_global_id(0) + id_0;
	kRap(k, x, y, z, ax, ay)
	dv[k] = ds[k] * Y[z];
}

kV BatchNormMediadnorm_norma(Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	int x, y;
	REAL md = 0;
	REAL m = 0;
	for (x = 0; x < ax; x++) {
		for (y = 0; y < ay; y++) {
			m += dv[kMap(x, y, z, ax, ay)];
			md += (dv[kMap(x, y, z, ax, ay)] * v[kMap(x, y, z, ax, ay)]);
		}
	}
	mdnorm[z] = m / (REAL) (ax * ay);
	mdnormnorm[z] = md / (REAL) (ax * ay);
}

kV BatchNormaCalcDa(Vr da, Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, Vr o, int ax, int ay, int id_0) {
	int x, y, z;
	int k = get_global_id(0) + id_0;
	kRap(k, x, y, z, ax, ay)
	da[k] = o[z] * (dv[k] - mdnorm[z] - v[k] * mdnormnorm[z]);
}

kV BatchNormaCalcdYdB(Vr ds, Vr v, Vw dY, Vw dB, long batchSize, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	REAL sumY = 0;
	REAL sumB = 0;
	int k;
	for (int x = 0; x < ax; ++x) {
		for (int y = 0; y < ay; ++y) {
			k = kMap(x, y, z, ax, ay);
			sumB += ds[k];
			sumY += ds[k] * v[k];
		}
	}
	dB[z] = dB[z] + sumB / (REAL) batchSize;
	dY[z] = dY[z] + sumY / (REAL) batchSize;
}

kV BatchNormaLearn(Vrw Y, Vrw B, Vrw dY, Vrw dB, REAL hit, REAL momento, REAL decaimento, int id_0) {
	int z = get_global_id(0) + id_0;
	Y[z] = Y[z] - hit * (dY[z] + Y[z] * decaimento);
	B[z] = B[z] - hit * (dB[z] + B[z] * decaimento);
	dY[z] = dY[z] * momento;
	dB[z] = dB[z] * momento;
}
//cnnutils.h
//
// Created by Henrique on 22-Jul-21.
//


kV createImg(__global unsigned char *out, Vr v, int vx, int vy, int imi, int imy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j, z;
	kRap(k, i, j, z, vx, vy)
	imi = imi + i;
	int imj = j + z * vy + z;
	out[imi * imy + imj] = ((int) v[k]) & 0xff;
}

kV putIMG(__global unsigned char *imagem_saida, Vr v, int z, REAL px, REAL py, int imy, int width, int i0, int j0, int vx, int vy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j;
	KRap2D(k, i, j, imy)
	int x = i * px, y = j * py;
	imagem_saida[(i + i0) * width + j + j0] = ((int) v[kMap(x, y, z, vx, vy)]) & 0xff;
}


kV normalizeVector(Vr input, Vr saida, REAL multiplicador, REAL somador, REAL subtrator, int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}


kV kernel_sub(Vr ds, Vr s, Vr t, int k0) {
	int k = get_global_id(0) + k0;
	ds[k] = s[k] - t[k];
}

kV kernel_normalizechar2real(Vr dst, __global unsigned char *src, REAL a, REAL b, int k0) {
	int k = get_global_id(0) + k0;
//	printf("update\n");
	dst[k] = ((REAL) src[k] - b) / a;
}

kV kernel_getVetorClassFromChar(Vr dst, __global unsigned char *ints, unsigned int noptiobs, int k0) {
	int w = get_global_id(0) + k0;
	int y = ints[w];
	dst[kMap4D(0, y, 0, w, 1, noptiobs, 1)] = 1.0;
}
kV kernel_fixW(Vr w, Vr dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0) {
	int k = get_global_id(0) + k0;
	w[k] = w[k] - hitlearn * (dw[k] + w[k] * decaimentoDePeso);
	dw[k] = dw[k] * momento;
}
//conv.h
kV convSum(Vr filtro, Vr entrada, Vr saida, int passox, int passoy, int saidatx, int saidaty, int entradatx, int entradaty, int fx, int fy, int fz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	kRap(k, x, y, filtrok, saidatx, saidaty)
	REAL sum = 0, f = 0, v = 0;
	int lf = 0, le = 0;
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			for (int z = 0; z < fz; z++) {
				lf = kMap4D(m, n, z, filtrok, fx, fy, fz);
				le = kMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);
				f = filtro[lf];
				v = entrada[le];
				sum += f * v;
			}
		}
	}
	saida[k] = sum;
}

kV convCalcGradAndFixWeight(Vr filtros, Vr ds, Vr entrada, Vr gradFiltro, int fx, int fy, int fz, int entrada_tx, int entrada_ty, int saida_tx, int saida_ty, int passox, int passoy, REAL hitLearn, REAL momento, REAL weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			le = kMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);
			ls = kMap(i, j, l, saida_tx, saida_ty);
			soma += entrada[le] * ds[ls];
		}
	}
	REAL dw = soma + gradFiltro[k] * momento;
	REAL w = filtros[k];
	filtros[k] = w - hitLearn * (dw + w * weightDecay);
	gradFiltro[k] = dw;
}

kV convCalcGradIn(Vr filtro, Vr gradEntrada, Vr gradNext, int fx, int fy, int fz, int passox, int passoy, int entradatx, int entradaty, int saidatx, int saidaty, int saidatz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, entradatx, entradaty)

	Range range_filtro;
	range_filtro.min.x = 0;
	if (x + fx > entradatx) {
		range_filtro.min.x = x + fx - entradatx;
	}
	range_filtro.max.x = fx - 1;
	if (x - fx + 1 < 0) {
		range_filtro.max.x = x;
	}
	range_filtro.min.y = 0;
	if (y + fy > entradaty) {
		range_filtro.min.y = y + fy - entradaty;
	}
	range_filtro.max.y = fy - 1;
	if (y - fy + 1 < 0) {
		range_filtro.max.y = y;
	}
	REAL somaErro = 0, pesoAplicado = 0;
	int i, j;
	int lf, ls;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / passox;
		if (i * passox + m != x) {
			continue;
		}
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / passoy;
			if (j * passoy + n != y) {
				continue;
			}
			for (int w = 0; w < saidatz; w++) {
				lf = kMap4D(m, n, z, w, fx, fy, fz);
				ls = kMap(i, j, w, saidatx, saidaty);
				pesoAplicado = filtro[lf];
				somaErro += pesoAplicado * gradNext[ls];
			}
		}
	}
	gradEntrada[k] = somaErro;
}

kV convCalcGradBatch(Vr ds, Vr entrada, Vr gradFiltro, long batchSize, int fx, int fy, int fz, int entrada_tx, int entrada_ty, int saida_tx, int saida_ty, int passox, int passoy, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			le = kMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);
			ls = kMap(i, j, l, saida_tx, saida_ty);
			soma += entrada[le] * ds[ls];
		}
	}
	soma = soma / batchSize + gradFiltro[k];
	gradFiltro[k] = soma;
}

//conv2d.h
kV conv2dSum(Vr W, Vr a, Vw Z, Vw s, int px, int py, int sx, int sy, int ax, int ay, int az, int fx, int fy, int fz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, zl;
	int z, l;
	kRap(k, x, y, zl, sx, sy)
	l = zl / az;
	z = zl % az;
	REAL sum = 0, f, v;
	int lf, le;
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			lf = kMap(m, n, l, fx, fy);
			le = kMap(x * px + m, y * py + n, z, ax, ay);
			f = W[lf];
			v = a[le];
			sum += f * v;
		}
	}
	Z[k] = sum;
	s[k] = func(fid, sum);
}

kV conv2dCalcGradZ(Vr ds, Vr z, Vw dz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	dz[k] = ds[k] * func(fid, z[k]);
}

kV conv2dCalcGradIn(Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int az, int sx, int sy,  int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, atx, aty)
	Range range_W;
	range_W.min.x = 0;
	if (x + fx > atx) {
		range_W.min.x = x + fx - atx;
	}
	range_W.max.x = fx - 1;
	if (x - fx + 1 < 0) {
		range_W.max.x = x;
	}
	range_W.min.y = 0;
	if (y + fy > aty) {
		range_W.min.y = y + fy - aty;
	}
	range_W.max.y = fy - 1;
	if (y - fy + 1 < 0) {
		range_W.max.y = y;
	}
	REAL somaErro = 0, pesoAplicado = 0;
	int i, j;
	int lf, ls;
	for (int m = range_W.min.x; m <= range_W.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x) {
			continue;
		}
		for (int n = range_W.min.y; n <= range_W.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y) {
				continue;
			}
			for (int l = 0; l < fz; l++) {
				lf = kMap(m, n, l, fx, fy);
				ls = kMap(i, j, l * az + z, sx, sy);
				pesoAplicado = W[lf];
				somaErro += pesoAplicado * dz[ls];
			}
		}
	}
	da[k] = somaErro;
}

kV conv2dCalcGradAndFixWeight(Vrw W, Vr dz, Vr a, Vrw dW, int fx, int fy, int ax, int ay, int az, int sx, int sy, int px, int py, REAL hitLearn, REAL momento, REAL weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, l;
	kRap(k, m, n, l, fx, fy)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < sx; ++i) {
		for (int j = 0; j < sy; ++j) {
			for (int z = 0; z < az; ++z) {
				ls = kMap(i, j, l * az + z, sx, sy);
				le = kMap(i * px + m, j * py + n, z, ax, ay);
				soma += a[le] * dz[ls];
			}
		}
	}
	REAL dw = soma + dW[k] * momento;
	REAL w = W[k];
	W[k] = w - hitLearn * (dw + w * weightDecay);
	dW[k] = dw;
}

kV conv2dCalcGradBatch(Vr dz, Vr a, Vr dW, long batchSize, int fx, int fy, int fz, int ax, int ay, int az, int sx, int sy, int px, int py, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, l;
	kRap(k, m, n, l, fx, fy)
	REAL soma = 0;
	int l_a, l_dz;
	for (int i = 0; i < sx; ++i) {
		for (int j = 0; j < sy; ++j) {
			for (int z = 0; z < az; ++z) {
				l_dz = kMap(i, j, l * az + z, sx, sy);
				l_a = kMap(i * px + m, j * py + n, z, ax, ay);
				soma += a[l_a] * dz[l_dz];
			}
		}
	}
	soma = soma / batchSize + dW[k];
	dW[k] = soma;
}


//convf.h
kV convFSum(Vr W, Vr a, Vw Z, Vw s, int px, int py, int sx, int sy, int atx, int aty, int fx, int fy, int fz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, Wk;

	kRap(k, x, y, Wk, sx, sy)
	REAL sum = 0, f, v;
	int lf, le;
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			for (int z = 0; z < fz; z++) {
				lf = kMap4D(m, n, z, Wk, fx, fy, fz);
				le = kMap(x * px + m, y * py + n, z, atx, aty);
				f = W[lf];
				v = a[le];
				sum += f * v;
			}
		}
	}
	Z[k] = sum;
	s[k] = func(fid, sum);
}

kV convFCalcGradZ(Vr ds, Vr z, Vw dz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	dz[k] = ds[k] * func(fid, z[k]);
}


kV convFCalcGradIn(Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int sx, int sy, int sz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, atx, aty)

	Range range_W;
	range_W.min.x = 0;
	if (x + fx > atx) {
		range_W.min.x = x + fx - atx;
	}
	range_W.max.x = fx - 1;
	if (x - fx + 1 < 0) {
		range_W.max.x = x;
	}
	range_W.min.y = 0;
	if (y + fy > aty) {
		range_W.min.y = y + fy - aty;
	}
	range_W.max.y = fy - 1;
	if (y - fy + 1 < 0) {
		range_W.max.y = y;
	}
	REAL somaErro = 0, pesoAplicado = 0;
	int i, j;
	int lf, ls;
	for (int m = range_W.min.x; m <= range_W.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x) {
			continue;
		}
		for (int n = range_W.min.y; n <= range_W.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y) {
				continue;
			}
			for (int w = 0; w < sz; w++) {
				lf = kMap4D(m, n, z, w, fx, fy, fz);
				ls = kMap(i, j, w, sx, sy);
				pesoAplicado = W[lf];
				somaErro += pesoAplicado * dz[ls];
			}
		}
	}
	da[k] = somaErro;
}

kV convFCalcGradAndFixWeight(Vr W, Vr dz, Vr a, Vr gradW, int fx, int fy, int fz, int a_tx, int a_ty, int s_tx, int s_ty, int px, int py, REAL hitLearn, REAL momento, REAL weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < s_tx; ++i) {
		for (int j = 0; j < s_ty; ++j) {
			le = kMap(i * px + m, j * py + n, z, a_tx, a_ty);
			ls = kMap(i, j, l, s_tx, s_ty);
			soma += a[le] * dz[ls];
		}
	}
	REAL dw = soma + gradW[k] * momento;
	REAL w = W[k];
	W[k] = w - hitLearn * (dw + w * weightDecay);
	gradW[k] = dw;
}

kV convFCalcGradBatch(Vr dz, Vr a, Vr dW, long batchSize, int fx, int fy, int fz, int a_tx, int a_ty, int s_tx, int s_ty, int px, int py, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < s_tx; ++i) {
		for (int j = 0; j < s_ty; ++j) {
			le = kMap(i * px + m, j * py + n, z, a_tx, a_ty);
			ls = kMap(i, j, l, s_tx, s_ty);
			soma += a[le] * dz[ls];
		}
	}
	soma = soma / batchSize + dW[k];
	dW[k] = soma;
}


//convNc.h
kV convncSum(Vr W, Vr A, Vr Z, Vr S, unsigned int fid, unsigned int passox, int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	kRap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d Kmapeado = {x * passox, y * passoy, 0};
	REAL sum = 0, f, v;
	for (int i = 0; i < fx; i++) {
		for (int j = 0; j < fy; j++) {
			for (int z = 0; z < fz; z++) {
				f = W[kMap4D(i, j, z, filtrok, fx, fy, fz)];
				v = A[kMap(Kmapeado.x + i * largx, Kmapeado.y + j * largy, z, entradatx, entradaty)];
				sum += f * v;
			}
		}
	}
	Z[k] = sum;
	S[k] = func(fid, sum);
}

kV convncCalcGradZ(Vr ds, Vr z, Vr dz, unsigned int fid, int k0) {
	int k = get_global_id(0) + k0;
	dz[k] = ds[k] * func(fid, z[k]);
}

kV convncCalcGrads(Vr W, Vr DA, Vr dz, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0) {
	/**
 * equacao a ser implementada \n
 * x = s*p + m*w \n
 * onde: \n
 * 	x é da entrada \n
 * 	s é da saida \n
 * 	m é do filtro\n
 * 	s = (x - m*w)/p \n
 */
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, entradatx, entradaty)
	Range range_filtro;
	range_filtro.min.x = 0;
	if ((entradatx - x - (fx - 1) * largx) < 0) {
		range_filtro.min.x = -entradatx + x + fx;
	}
	range_filtro.max.x = fx - 1;
	if (x - (fx - 1) * largx < 0) {
		range_filtro.max.x = x / largx;
	}
	range_filtro.min.y = 0;
	if ((entradaty - y - (fy - 1) * largy) < 0) {
		range_filtro.min.y = -entradaty + y + fy;
	}
	range_filtro.max.y = fy - 1;
	if (y - (fy - 1) * largy < 0) {
		range_filtro.max.y = y / largy;
	}
	int sx, sy;
	REAL somaErro = 0, aux, pesoAplicado = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		sx = (x - m * largx) / passox;
		if (sx * passox + m * largx != x) {
			continue;
		}
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			sy = (y - n * largy) / passox;
			if (sy * passoy + n * largy != y) {
				continue;
			}
			for (int l = 0; l < fz; l++) {
				pesoAplicado = W[kMap4D(m, n, z, l, fx, fy, fz)];
				aux = pesoAplicado * dz[kMap(sx, sy, l, saidatx, saidaty)];
				somaErro += aux;
			}
		}
	}
	DA[k] = somaErro;
}

kV convncCalcFiltro(Vr dz, Vr A, Vr W, Vr dW, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, REAL hitlearn, REAL momento, REAL weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, dw_x, dw_y, dw_z)
	REAL soma = 0, aux;
	for (int i = 0; i < s_x; ++i) {
		for (int j = 0; j < s_y; ++j) {
			aux = A[kMap(i * passox + m * largx, j * passoy + n * largy, z, a_x, a_y)] * dz[kMap(i, j, l, s_x, s_y)];
			soma += aux;
		}
	}
	dW[k] = soma + dW[k] * momento;
	W[k] = W[k] - hitlearn * (dW[k] + W[k] * weightDecay);
}
kV convncCalcFiltroBatch(Vr dz, Vr A, Vr dW, long batchSize, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, dw_x, dw_y, dw_z)
	REAL soma = 0, aux;
	for (int i = 0; i < s_x; ++i) {
		for (int j = 0; j < s_y; ++j) {
			aux = A[kMap(i * passox + m * largx, j * passoy + n * largy, z, a_x, a_y)] * dz[kMap(i, j, l, s_x, s_y)];
			soma += aux;
		}
	}
	dW[k] = soma/batchSize;
}




//dropout.h
#define MAX_INT_DP  ((1UL << 31) - 1)

long randoml(unsigned long seed, unsigned long id) {
	seed += id;
	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;
}

REAL randomD(unsigned long seed, unsigned long id) {
	return (REAL) randoml(seed, id) / (REAL) MAX_INT_DP;
}

kV dropativa(Vr entrada, Vr saida, __global char *hitmap, long seed, REAL pativa, int k0) {
	int i = get_global_id(0) + k0;
//	printf("kernel %lf %lf %g %g\n",randomD(seed, i),pativa,(REAL)(seed +i),(REAL)MAX_INT_DP);
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i];
}


kV dropcalcgrad(Vr gradentrada, __global char *hitmap, Vr gradnext, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i];
}

//fullconnect.h
kV fullfeed(Vr a, Vr w, Vr b, Vr z, Vr s, int fid, int w_x, int w_y, int k0) {
	int m = get_global_id(0) + k0;
	REAL sum = 0;
	int n;
	for (n = 0; n < w_y; n++) {
		sum += a[n] * w[kMap(m, n, 0, w_x, w_y)];
	}
	z[m] = sum + b[m];
	s[m] = func(fid, z[m]);
}

kV fullCalcDWandFix(Vr a, Vr w, Vr dw, Vr dz, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int pesosy, int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] + dw[k] * momento;
	w[k] = w[k] - hitlearn * (dw[k] + w[k] * decaimentoDePeso);
}


kV fullCalcDz(Vr dz, Vr ds, Vr z, int dfa, int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
}
kV fullCalcDzBath(Vr dz, Vr ds, Vr z, Vr db, int dfa, long batchSize, int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
	db[m] = dz[m]/batchSize + db[m];
}

kV fullCalcDzAndFixB(Vr dz, Vr ds, Vr z, Vr b, Vr db, int dfa, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
	db[m] = dz[m] + db[m] * momento;
	b[m] = b[m] - hitlearn * (db[m] + b[m] * decaimentoDePeso);
}


kV fullcalcin(Vr dz, Vr da, Vr w, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	REAL soma = 0;
	for (int n = 0; n < pesosx; ++n) {
		soma += dz[n] * w[kMap(n, m, 0, pesosx, pesosy)];
	}
	da[m] = soma;
}


kV fullCalcDWBatch(Vr a, Vr dw, Vr dz, long batchSize, int pesosy, int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] / batchSize + dw[k];
}


//padding.h
kV paddingfeed(Vr in, Vr out, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, txi, tyi)
	int s = kMap(x + t, y + l, z, txo, tyo);
	out[s] = in[k];
}

kV paddingBack(Vr gradNext, Vr gradin, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, txi, tyi)
	int s = kMap(x + t, y + l, z, txo, tyo);
	gradin[k] = gradNext[s];
}
//poolav.h
kV poolAVativa(Vr entrada, Vr saida, int passox, int passoy, int fx, int fy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL soma = 0;

	for (int i = 0; i < fx; ++i) {
		for (int j = 0; j < fy; ++j) {
			soma += entrada[kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
		}
	}
	saida[k] = soma / (fx * fy);
}


kV poolAvCalcGrads(Vr entrada, Vr gradEntrada, Vr gradNext, Vr saida, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, entradatx, entradaty)
	Range range_filtro;
	range_filtro.min.x = 0;
	if (x + fx > entradatx) {
		range_filtro.min.x = x + fx - entradatx;
	}
	range_filtro.max.x = fx - 1;
	if (x - fx + 1 < 0) {
		range_filtro.max.x = x;
	}
	range_filtro.min.y = 0;
	if (y + fy > entradaty) {
		range_filtro.min.y = y + fy - entradaty;
	}
	range_filtro.max.y = fy - 1;
	if (y - fy + 1 < 0) {
		range_filtro.max.y = y;
	}
	int i, j;//saida
	REAL soma = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x) {
			continue;
		}
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y) {
				continue;
			}
			soma += gradNext[kMap(i, j, z, saidatx, saidaty)];
		}
	}
	gradEntrada[kMap(x, y, z, entradatx, entradaty)] = soma / (fx * fy);

}


//poolMax.h
kV poolativa(Vr entrada, Vr saida, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = -DBL_MAX;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			v = entrada[kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
			if (v > mval) {
				mval = v;
			}
		}
	}
	saida[k] = mval;
}


kV poolCalcGrads(Vr entrada, Vr gradEntrada, Vr gradNext, Vr saida, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, entradatx, entradaty)
	Range range_filtro;
	if (x + fx > entradatx) {
		range_filtro.min.x = x + fx - entradatx;
	}
	range_filtro.max.x = fx - 1;
	if (x - fx + 1 < 0) {
		range_filtro.max.x = x;
	}
	range_filtro.min.y = 0;
	if (y + fy > entradaty) {
		range_filtro.min.y = y + fy - entradaty;
	}
	range_filtro.max.y = fy - 1;
	if (y - fy + 1 < 0) {
		range_filtro.max.y = y;
	}
	int i, j;//saida
	gradEntrada[kMap(x, y, z, entradatx, entradaty)] = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x) {
			continue;
		}
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y) {
				continue;
			}
			if (entrada[k] == saida[kMap(i, j, z, saidatx, saidaty)]) {
				gradEntrada[k] = gradNext[kMap(i, j, z, saidatx, saidaty)];
				return;
			}
		}
	}

}


//poolMin.h
kV poolativaMin(Vr entrada, Vr saida, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = DBL_MAX;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			v = entrada[kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
			if (v < mval) {
				mval = v;
			}
		}
	}
	saida[k] = mval;
}


//prelu.h
kV preluativa(Vr entrada, Vr saida, Vr A, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		v = v * A[k];
	}
	saida[k] = v;
}

kV prelucalcgrad(Vr gradentrada, Vr entrada, Vr gradnext, Vr A, Vr dA, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		gradentrada[k] = gradnext[k] * A[k];
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		gradentrada[k] = gradnext[k];
		dA[k] = momento * dA[k];
	}
	if (learn) {
		A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
	}
}

kV preluonlyfix(Vr entrada, Vr gradnext, Vr A, Vr dA, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		dA[k] = momento * dA[k];
	}
	A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
}

kV prelucalcgradBatch(Vr gradentrada, Vr entrada, Vr gradnext, Vr A, Vr dA, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		gradentrada[k] = gradnext[k] * A[k];
		dA[k] = gradnext[k] / batchSize + dA[k];
	} else {
		gradentrada[k] = gradnext[k];
		dA[k] = 1.0 / batchSize + dA[k];
	}
}

kV preluonlyDABatch(Vr entrada, Vr gradnext, Vr A, Vr dA, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		dA[k] = gradnext[k] / batchSize + dA[k];
	} else {
		dA[k] = 1.0 / batchSize + dA[k];
	}
}
//relu.h

kV reluativa(Vr entrada, Vr saida, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = entrada[k] < 0.0 ? (entrada[k] * menor) : (entrada[k] * maior);
}


kV relucalcgrad(Vr gradentrada, Vr entrada, Vr gradnext, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	gradentrada[k] = entrada[k] < 0.0 ? (menor * gradnext[k]) : (maior * gradnext[k]);
}

//softmax.h
/**
 * @goal calcular e^a(x,y,z)
 * @iteration dimensão de a (x,y,z)
 * @param entrada Tensor de entrada (leitura)
 * @param exponent Tensor e^entrada (escrita)
 * @param k0 usado internamente no kernel
 */
kV softmaxExp(Vr entrada, Vr exponent, int k0) {
	int k = get_global_id(0) + k0;
	exponent[k] = EXP(entrada[k]);
}

/***
 * @goal encontrar a soma de cada dimensão z
 * @iteration dimensão z da entrada a(:,:,z)
 * @param eps Tensor exponencial da entrada (leitura)
 * @param soma Tensor da soma das exponenciais (escrita)
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão x da saída
 * @param k0 usado internamente no kernel
 */
kV softmaxSomaExp(Vr eps, Vr soma, int saidatx, int saidaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL sum = 0;
	for (x = 0; x < saidatx; x++) {
		for (y = 0; y < saidaty; y++) {
			sum += eps[kMap(x, y, z, saidatx, saidaty)];
		}
	}
	soma[z] = sum;
}
/***
 * @goal Normalizar a exponencial pela soma
 *  * @iteration dimensão da saída  s(x,y,z)
 * @param exponet Tensor exponencial da entrada (leitura)
 * @param soma Tensor da soma das exponenciais (leitura)
 * @param saida Tensor de saída (escrita)
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão x da saída
 * @param k0 usado internamente no kernel
 */
kV softmaxNormaliza(Vr exponet, Vr soma, Vr saida, int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)
	saida[k] = exponet[k] / soma[z];
}
/**
 * @goal Calcular os gradientes de entrada
 * @iteration dimensão da entrada a(x,y,z)
 * @param da Tensor de gradientes de entrada (escrita)
 * @param s Tensor de saida (leitura)
 * @param ds Tensor gradiente da saída (leitura)
 * @param sx dimensão x da saída
 * @param sy dimensão y da saída
 * @param k0 usado internamente no kernel
 */
kV softMaxcalcgrad(Vr da, Vr s, Vr ds, int sx, int sy, int k0) {
	int k = get_global_id(0) + k0;
	int i, z, j;
	int sxy = sx * sy;
	KRap2D(k, z, i, sxy);
	REAL yi = s[k];
	REAL soma = 0.0;
	for (j = 0; j < sxy; ++j) {
		if (j == i) {
			soma += yi * (1 - yi) * ds[j + z * sxy];
//			printf("v(%d,%d,%d) =  %f, %f %f;\n", i+1, j+1, z+1, yi * (1 - yi),s[j + z * sxy],yi);
		} else {
			soma += -yi * s[j + z * sxy] * ds[j + z * sxy];
//			printf("v(%d,%d,%d) =  %f, %f %f;\n", i+1, j+1, z+1, yi * -s[j + z * sxy],s[j + z * sxy],yi);
		}
	}
	da[k] = soma;
}

/**
 * @goal Encontrar o maximo e o indice de cada dimensão z
 * @iteration dimensão z da entrada a(:,:,z)
 * @param a entrada
 * @param mx tensor maximos
 * @param i_max tensor indice de maximos
 * @param ax entrada x
 * @param ay entrada y
 * @param k0 uso interno no kernel
 */
kV softmaxFindMax(Vr a, Vr mx, __global int *i_max, int ax, int ay, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL maximo = a[kMap(0, 0, z, ax, ay)];
	REAL adata;
	int imax = 0;
	for (x = 0; x < ax; x++) {
		for (y = 0; y < ay; y++) {
			adata = a[kMap(x, y, z, ax, ay)];
			if (maximo < adata) {
				maximo = adata;
				imax = x * ay + y;
			}
		}
	}
	i_max[z] = imax;
	mx[z] = maximo;
}

/**
 * @goal calcular e^(a(x,y,z) - max(a))
 * @iteration dimensão de a (x,y,z)
 * @param entrada Tensor de entrada (leitura)
 * @param exponent Tensor e^entrada (escrita)
 * @param mx tensor maximos
 * @param ax entrada x
 * @param ay entrada y
 * @param k0 usado internamente no kernel
 */
kV softmaxExpNorm(Vr entrada, Vr exponent, Vr mx, int ax, int ay, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, ax, ay);
	exponent[k] = EXP(entrada[k] - mx[z]);
}

#endif //GAB_KERNELS_OPENCL_H
