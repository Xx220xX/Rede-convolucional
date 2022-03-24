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
	w[k] = w[k] - hitlearn * dw[k] -  hitlearn * w[k] * decaimentoDePeso ;
	dw[k] = dw[k] * momento;
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

kV dropativaTreino(Vr entrada, Vw saida, __global char *hitmap, long seed, REAL pativa, int k0) {
	int i = get_global_id(0) + k0;
//	printf("kernel %lf %lf %g %g\n",randomD(seed, i),pativa,(REAL)(seed +i),(REAL)MAX_INT_DP);
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i]/pativa;
}
kV dropativaPredict(Vr entrada, Vw saida, REAL pativa, int k0) {
	int i = get_global_id(0) + k0;
	saida[i] = entrada[i] ;//* pativa;
}

kV dropcalcgrad(Vr gradentrada, __global char *hitmap, Vr gradnext, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i];
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


kV poolAvCalcGrads(Vr A, Vw dA, Vr dS, Vr S, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0) {
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
			soma += dS[kMap(i, j, z, saidatx, saidaty)];
		}
	}
	dA[kMap(x, y, z, entradatx, entradaty)] = soma / (fx * fy);

}


//poolMax.h
kV poolativa(Vr entrada, Vw saida,Vw hmap, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = entrada[kMap(mapeado.x , mapeado.y , z, entradatx, entradaty)];
	int mx = 0;
	int index;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			index = kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty);
			v = entrada[index];
			if (v > mval) {
				mval = v;
				mx = index;
			}
		}
	}
	saida[k] = mval;
	hmap[k] = mx;
}


kV poolCalcGrads(Vr A, Vr dA, Vr dS,  __global int * hmap, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0) {
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
			if (k == hmap[kMap(i, j, z, saidatx, saidaty)]) {
				soma += dS[kMap(i, j, z, saidatx, saidaty)];
			}
		}
	}
	dA[k] = soma;
}


//poolMin.h
kV poolativaMin(Vr A, Vr S, __global int * hmap, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = A[kMap(mapeado.x , mapeado.y , z, entradatx, entradaty)];
	int index,mn=0;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			index = kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty);
			v = A[index];
			if (v < mval) {
				mval = v;
				mn = index;
			}
		}
	}
	S[k] = mval;
	hmap[k] = mn;
}


//prelu.h
kV preluativa(Vr A, Vw S, Vr W, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		v = v * W[k];
	}
	S[k] = v;
}

kV prelucalcgrad(Vw dA, Vr A, Vr dS, Vrw W, Vrw dW, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dA[k] = dS[k] * W[k];
		dW[k] = dS[k] + momento * dW[k];
	} else {
		dA[k] = dS[k];
		dW[k] = momento * dW[k];
	}
	if (learn) {
		W[k] = W[k] - hitlearn * (dW[k] + W[k] * decaimento);
	}
}

kV preluonlyfix(Vr A, Vr dS, Vrw W, Vrw dW, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dW[k] = dS[k] + momento * dW[k];
	} else {
		dW[k] = momento * dW[k];
	}
	W[k] = W[k] - hitlearn * (dW[k] + W[k] * decaimento);
}

kV prelucalcgradBatch(Vw dA, Vr A, Vr dS, Vr W, Vrw dW, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dA[k] = dS[k] * W[k];
		dW[k] = dS[k] / batchSize + dW[k];
	} else {
		dA[k] = dS[k];
		dW[k] = 1.0 / batchSize + dW[k];
	}
}

kV preluonlyDABatch(Vr A, Vr dS, Vr W, Vr dW, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dW[k] = dS[k] / batchSize + dW[k];
	} else {
		dW[k] = 1.0 / batchSize + dW[k];
	}
}
//relu.h

kV reluativa(Vr A, Vr S, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	S[k] = A[k] < 0.0 ? (A[k] * menor) : (A[k] * maior);
}


kV relucalcgrad(Vr dA, Vr A, Vr dS, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	dA[k] = A[k] < 0.0 ? (menor * dS[k]) : (maior * dS[k]);
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
