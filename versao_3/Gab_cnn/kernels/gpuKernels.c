#ifndef GAB_KERNELS_OPENCL_H
#define GAB_KERNELS_OPENCL_H
//utils.h
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
//bathnorm.h

/// achar a media
/// ativa 1
kV BatchNormMedia(Vector entrada, Vector media,
				  int entradatx, int entradaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL m = 0;
	for (x = 0; x < entradatx; x++) {
		for (y = 0; y < entradaty; y++) {
			m += entrada[KTensorMap(x, y, z, entradatx, entradaty)];
		}
	}
	media[z] = m / (REAL)(entradatx * entradaty);
}

/// achar a diferenca
/// ativa 2
kV BatchNormDiferenca(Vector entrada, Vector media,
					  Vector diferenca,
					  Vector diferencaquad,
					  int entradatx, int entradaty, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, entradatx, entradaty)
	diferenca[k] = entrada[k] - media[z];
	diferencaquad[k] = diferenca[k] * diferenca[k];
}
/// ativa 3

kV BatchNormVariance(Vector dif, Vector difQuad,
					 Vector sumdiferenca, Vector variancia,
					 REAL episolon, int diftx, int difty,
					 int k0) {
	int z = get_global_id(0) + k0;
	REAL sum = 0;
	REAL sumdif = 0;
	for (int x = 0; x < diftx; x++) {
		for (int y = 0; y < difty; y++) {
			sum += difQuad[KTensorMap(x, y, z, diftx, difty)];
			sumdif += dif[KTensorMap(x, y, z, diftx, difty)];
		}
	}
	sumdiferenca[z] = sumdif;
	variancia[z] = SQRT(sum / (difty * diftx) + episolon);
}

/// normaliza
/// ativa 4

kV BatchNormNormaliza(Vector saida,
					  Vector norma,
					  Vector diferenca,
					  Vector variancia,
					  Vector Y,
					  Vector B,
					  int diferencatx, int diferencaty, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, diferencatx, diferencaty)
	norma[k] = diferenca[k] / variancia[z];
	saida[k] = norma[k] * Y[z] + B[z];
}


kV BatchNormaCalcGrad1(Vector gradIn,
					   Vector gradNext,
					   Vector variancia,
					   Vector media,
					   Vector Y,
					   Vector somaDif,
					   Vector entrada,
					   int entradatx,
					   int entradaty,
					   int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, entradatx, entradaty)
	REAL M = entradatx * entradaty;
	REAL dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);
	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);

	REAL didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;
	didx = didx / (variancia[z] * variancia[z]);
	didx = didx * gradNext[k];
	gradIn[k] = didx * Y[z];
}
kV BatchNormaCalcGrad2(Vector gradNext,
					   Vector norma,
					   Vector Y,
					   Vector B,
					   Vector gradY,
					   Vector gradB,
					   REAL hitlearn,
					   REAL momento,
					   REAL weightDecay,
					   int entradatx,
					   int entradaty,
					   int k0) {
	int z = get_global_id(0) + k0;
	REAL sumY = 0;
	REAL sumB = 0;
	int k;
	for (int x = 0; x < entradatx; ++x) {
		for (int y = 0; y < entradaty; ++y) {
			k = KTensorMap(x, y, z, entradatx, entradaty);
			sumY += gradNext[k];
			sumB += gradNext[k] * norma[k];
		}
	}
	gradB[z] = sumB + gradB[z] * momento;
	gradY[z] = sumY + gradY[z] * momento;

	B[z] = B[z] - (gradB[z] + weightDecay * B[z]) * hitlearn;
	Y[z] = Y[z] - (gradY[z] + weightDecay * Y[z]) * hitlearn;
}



//cnnutils.h
//
// Created by Henrique on 22-Jul-21.
//


kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j, z;
	KTensorRemap(k, i, j, z, vx, vy)
	imi = imi + i;
	int imj = j + z * vy + z;
	out[imi * imy + imj] = ((int) v[k]) & 0xff;
}

kV putIMG(__global unsigned char *imagem_saida,
		  Vector v,
		  int z,
		  REAL px,
		  REAL py,
		  int imy,
		  int width,
		  int i0,
		  int j0,
		  int vx,
		  int vy,
		  int k0) {
	int k = get_global_id(0) + k0;
	int i, j;
	KTensorRemap2D(k, i, j, imy)
	int x = i * px, y = j * py;
	imagem_saida[(i + i0) * width + j + j0] = ((int) v[KTensorMap(x, y, z, vx, vy)]) & 0xff;
}


kV normalizeVector(Vector input, Vector saida, REAL multiplicador, REAL somador, REAL subtrator,
				   int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}


kV kernel_sub(Vector ds, Vector s, Vector t, int k0) {
	int k = get_global_id(0) + k0;
	ds[k] = s[k] - t[k];
}

kV kernel_normalizechar2real(Vector dst, __global char *src, REAL a, REAL b, int k0) {
	int k = get_global_id(0) + k0;
	dst[k] = ((REAL)src[k] - b) / a;
}

kV kernel_getVetorClassFromChar( Vector dst, __global unsigned char *ints,int noptiobs, int k0) {
	int w = get_global_id(0) + k0;
	int y = ints[w];
	dst[KTensorMap4D(0, y, 0, w, 1, noptiobs, 1)] = 1.0;
}

//conv.h
kV convSum(Vector filtro, Vector entrada, Vector saida,
           int passox, int passoy,
           int saidatx, int saidaty,
           int entradatx, int entradaty,
           int fx, int fy, int fz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	KTensorRemap(k, x, y, filtrok, saidatx, saidaty)
	REAL sum = 0, f = 0, v = 0;
	int lf = 0, le = 0;
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			for (int z = 0; z < fz; z++) {
				lf = KTensorMap4D(m, n, z, filtrok, fx, fy, fz);
				le = KTensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);
				f = filtro[lf];
				v = entrada[le];
				sum += f * v;
			}
		}
	}
	saida[k] = sum;
}


kV convCalcGradAndFixWeight(Vector filtros, Vector ds,
                            Vector entrada, Vector gradFiltro,
                            int fx, int fy, int fz,
                            int entrada_tx, int entrada_ty,
                            int saida_tx, int saida_ty,
                            int passox, int passoy,
                            REAL hitLearn, REAL momento, REAL weightDecay,
                            int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	KTensorRemap4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			le = KTensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);
			ls = KTensorMap(i, j, l, saida_tx, saida_ty);
			soma += entrada[le]
			        * ds[ls];
		}
	}
	REAL dw = soma + gradFiltro[k] * momento;
	REAL w = filtros[k];
	filtros[k] = w - hitLearn * (dw + w * weightDecay);
	gradFiltro[k] = dw;
}

kV convCalcGradIn(Vector filtro, Vector gradEntrada, Vector gradNext,
                  int fx, int fy, int fz,
                  int passox, int passoy,
                  int entradatx, int entradaty,
                  int saidatx, int saidaty, int saidatz,
                  int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, entradatx, entradaty)

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
		if (i * passox + m != x) continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / passoy;
			if (j * passoy + n != y) continue;
			for (int w = 0; w < saidatz; w++) {
				lf = KTensorMap4D(m, n, z, w, fx, fy, fz);
				ls = KTensorMap(i, j, w, saidatx, saidaty);
				pesoAplicado = filtro[lf];
				somaErro += pesoAplicado * gradNext[ls];
			}
		}
	}
	gradEntrada[k] = somaErro;
}


//convf.h
kV convFSum(Vector filtro, Vector entrada, Vector Z, Vector saida,
			int passox, int passoy,
			int saidatx, int saidaty,
			int entradatx, int entradaty,
			int fx, int fy, int fz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;

	KTensorRemap(k, x, y, filtrok, saidatx, saidaty)
	REAL sum = 0, f, v;
	int lf, le;
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			for (int z = 0; z < fz; z++) {
				lf = KTensorMap4D(m, n, z, filtrok, fx, fy, fz);
				le = KTensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);
				f = filtro[lf];
				v = entrada[le];
				sum += f * v;
			}
		}
	}
	Z[k] = sum;
	saida[k] = func(fid, sum);
}

kV convFCalcGradZ(Vector ds, Vector z, Vector dz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	dz[k] = ds[k] * func(fid, z[k]);
}



kV convFCalcGradIn(Vector filtro, Vector gradEntrada, Vector dz,
				   int fx, int fy, int fz,
				   int passox, int passoy,
				   int entradatx, int entradaty,
				   int saidatx, int saidaty, int saidatz,
				   int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, entradatx, entradaty)

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
		if (i * passox + m != x) continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / passoy;
			if (j * passoy + n != y) continue;
			for (int w = 0; w < saidatz; w++) {
				lf = KTensorMap4D(m, n, z, w, fx, fy, fz);
				ls = KTensorMap(i, j, w, saidatx, saidaty);
				pesoAplicado = filtro[lf];
				somaErro += pesoAplicado * dz[ls];
			}
		}
	}
	gradEntrada[k] = somaErro;
}

kV convFCalcGradAndFixWeight(Vector filtros, Vector dz,
							 Vector entrada, Vector gradFiltro,
							 int fx, int fy, int fz,
							 int entrada_tx, int entrada_ty,
							 int saida_tx, int saida_ty,
							 int passox, int passoy,
							 REAL hitLearn, REAL momento, REAL weightDecay,
							 int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	KTensorRemap4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			le = KTensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);
			ls = KTensorMap(i, j, l, saida_tx, saida_ty);
			soma += entrada[le]
					* dz[ls];
		}
	}
	REAL dw = soma + gradFiltro[k] * momento;
	REAL w = filtros[k];
	filtros[k] = w - hitLearn * (dw + w * weightDecay);
	gradFiltro[k] = dw;
}
//convNc.h

kV convncSum(Vector W, Vector A, Vector Z, Vector S,
			 unsigned int fid,
			 unsigned int passox, int passoy,
			 unsigned int largx, unsigned int largy,
			 unsigned int entradatx, unsigned int entradaty,
			 unsigned int saidatx, unsigned int saidaty,
			 unsigned int fx, unsigned int fy, unsigned int fz,
			 int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	KTensorRemap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d Kmapeado = {x * passox, y * passoy, 0};
	REAL sum = 0, f, v;
	for (int i = 0; i < fx; i++)
		for (int j = 0; j < fy; j++)
			for (int z = 0; z < fz; z++) {
				f = W[KTensorMap4D(i, j, z, filtrok, fx, fy, fz)];
				v = A[KTensorMap(Kmapeado.x + i * largx, Kmapeado.y + j * largy, z, entradatx, entradaty)];

				sum += f * v;
			}
	Z[k] = sum;
	S[k] = func(fid, sum);
}

kV convncCalcGradZ(Vector ds, Vector z, Vector dz, unsigned int fid, int k0) {
	int k = get_global_id(0) + k0;
	dz[k] = ds[k] * func(fid, z[k]);
}

kV convncCalcGrads(Vector W,
				   Vector DA,
				   Vector dz,
				   unsigned int passox, unsigned int passoy,
				   unsigned int largx, unsigned int largy,
				   unsigned int entradatx, unsigned int entradaty,
				   unsigned int saidatx, unsigned int saidaty,
				   unsigned int fx, unsigned int fy, unsigned int fz,
				   int k0) {
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
	KTensorRemap(k, x, y, z, entradatx, entradaty)
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
		if (sx * passox + m * largx != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			sy = (y - n * largy) / passox;
			if (sy * passoy + n * largy != y)continue;
			for (int l = 0; l < fz; l++) {
				pesoAplicado = W[KTensorMap4D(m, n, z, l, fx, fy, fz)];
				aux = pesoAplicado * dz[KTensorMap(sx, sy, l, saidatx, saidaty)];
				somaErro += aux;
			}
		}
	}
	DA[k] = somaErro;
}

kV convncCalcFiltro(Vector dz,
					Vector A,
					Vector W,
					Vector dW,
					unsigned int dw_x, unsigned int dw_y, unsigned int dw_z,
					unsigned int a_x, unsigned int a_y,
					unsigned int s_x, unsigned int s_y,
					unsigned int passox, unsigned int passoy,
					unsigned int largx, unsigned int largy,
					REAL hitlearn, REAL momento, REAL weightDecay,
					int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	KTensorRemap4D(k, m, n, z, l, dw_x, dw_y, dw_z)
	REAL soma = 0, aux;
	for (int i = 0; i < s_x; ++i) {
		for (int j = 0; j < s_y; ++j) {
			aux = A[KTensorMap(i * passox + m * largx, j * passoy + n * largy, z, a_x, a_y)]
				  * dz[KTensorMap(i, j, l, s_x, s_y)];
			soma += aux;
		}
	}
	dW[k] = soma + dW[k] * momento;
	W[k] = W[k] - hitlearn * (dW[k] + W[k] * weightDecay);

}




//dropout.h
#define MAX_INT_DP  ((1UL << 31) - 1)
long randoml(unsigned long seed,unsigned long id) {
	seed += id;
	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;
}

REAL randomD(unsigned long seed,unsigned long id) {
	return (REAL) randoml(seed, id) / (REAL) MAX_INT_DP;
}

kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,
			 REAL pativa, int k0) {
	int i = get_global_id(0) + k0;
//	printf("kernel %lf %lf %g %g\n",randomD(seed, i),pativa,(REAL)(seed +i),(REAL)MAX_INT_DP);
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i];
}


kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i];
}

//fullconnect.h

/**
 *
 * @param entrada (N,1,1)
 * @param pesos (M,N,1)
 * @param b (N,1,1)
 * @param z (N,1,1)
 * @param saida (N,1,1)
 * @param funcaoativacao
 * @param pesosx M
 * @param pesosy N
 * @param k0
 */
kV fullfeed(Vector entrada, Vector pesos, Vector b, Vector z, Vector saida,
			int funcaoativacao, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	REAL valorEntrada = 0;
	int n;
	for (n = 0; n < pesosy; n++) {
		valorEntrada += entrada[n] * pesos[KTensorMap(m, n, 0, pesosx, pesosy)];
	}
	z[m] = valorEntrada + b[m];
	saida[m] = func(funcaoativacao, valorEntrada);
}

kV fullCalcDWandFix(Vector a,
					Vector w,
					Vector dw,
					Vector dz,
					REAL hitlearn,
					REAL momento,
					REAL decaimentoDePeso,
					int pesosy,
					int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] + dw[k] * momento;
	w[k] = w[k] - hitlearn * (dw[k] + w[k] * decaimentoDePeso);
}

kV fullCalcDz(Vector dz, Vector ds, Vector z, Vector b, Vector db,
			  int dfa, REAL hitlearn,
			  REAL momento, REAL decaimentoDePeso,
			  int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
}

kV fullCalcDzAndFixB(Vector dz, Vector ds, Vector z, Vector b,
					 Vector db, int dfa, REAL hitlearn,
					 REAL momento, REAL decaimentoDePeso,
					 int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
	db[m] = dz[m] + db[m] * momento;
	b[m] = b[m] - hitlearn * (db[m] + b[m] * decaimentoDePeso);
}

kV fullcalcin(Vector dz, Vector da, Vector w, int pesosx, int pesosy,
			  int k0) {
	int m = get_global_id(0) + k0;
	REAL soma = 0;
	for (int n = 0; n < pesosx; ++n) {
		soma += dz[n] * w[KTensorMap(n, m, 0, pesosx, pesosy)];
	}
	da[m] = soma;
}

//padding.h

kV paddingfeed(Vector in,Vector out,
			   unsigned int txi,unsigned int tyi,
			   unsigned int txo,unsigned int tyo,
			   unsigned int t, unsigned int l ,
			   int k0){
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, txi, tyi)
	int s = KTensorMap(x+t,y+l,z,txo,tyo);
	out[s] = in[k];
}
kV paddingBack(Vector gradNext,Vector gradin,
			   unsigned int txi, unsigned int tyi,
			   unsigned int txo,unsigned int tyo,
			   unsigned int t, unsigned int l , int k0){
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, txi, tyi)
	int s = KTensorMap(x+t,y+l,z,txo,tyo);
	gradin[k] = gradNext[s];
}
//poolav.h
kV poolAVativa(Vector entrada, Vector saida,
			   int passox, int passoy,
			   int fx, int fy,
			   int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL soma = 0;

	for (int i = 0; i < fx; ++i) {
		for (int j = 0; j < fy; ++j) {
			soma += entrada[KTensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
		}
	}
	saida[k] = soma / (fx * fy);
}


kV poolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,
				   int fx, int fy, int px, int py,
				   int entradatx, int entradaty,
				   int saidatx, int saidaty,
				   int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, entradatx, entradaty)
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
		if (i * px + m != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y)continue;
			soma += gradNext[KTensorMap(i, j, z, saidatx, saidaty)];
		}
	}
	gradEntrada[KTensorMap(x, y, z, entradatx, entradaty)] = soma / (fx * fy);

}


//poolMax.h
kV poolativa(Vector entrada, Vector saida,
			 int passox, int passoy,
			 int filtrox, int filtroy,
			 int saidatx, int saidaty,
			 int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = -DBL_MAX;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			v = entrada[KTensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
			if (v > mval)
				mval = v;
		}
	}
	saida[k] = mval;
}


kV poolCalcGrads(Vector entrada, Vector gradEntrada,
				 Vector gradNext, Vector saida,
				 int fx, int fy, int px, int py,
				 int entradatx, int entradaty,
				 int saidatx, int saidaty,
				 int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, entradatx, entradaty)
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
	gradEntrada[KTensorMap(x, y, z, entradatx, entradaty)] = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y)continue;
			if (entrada[k] == saida[KTensorMap(i, j, z, saidatx, saidaty)]) {
				gradEntrada[k] = gradNext[KTensorMap(i, j, z, saidatx, saidaty)];
				return;
			}
		}
	}

}


//poolMin.h
kV poolativaMin(Vector entrada, Vector saida,
				int passox, int passoy,
				int filtrox, int filtroy,
				int saidatx, int saidaty,
				int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = DBL_MAX;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			v = entrada[KTensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
			if (v < mval)
				mval = v;
		}
	}
	saida[k] = mval;
}


//prelu.h
kV preluativa(Vector entrada, Vector saida, Vector A, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0)
		v = v * A[k];
	saida[k] = v;
}

kV prelucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA,
				 int learn, REAL hitlearn, REAL momento,
				 REAL decaimento,
				 int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		gradentrada[k] = gradnext[k] * A[k];
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		gradentrada[k] = gradnext[k];
		dA[k] = momento * dA[k];
	}
	if (learn)
		A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
}

kV preluonlyfix(Vector entrada, Vector gradnext, Vector A, Vector dA,
				REAL hitlearn, REAL momento,
				REAL decaimento,
				int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		dA[k] = momento * dA[k];
	}
	A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
}

//relu.h

kV reluativa(Vector entrada, Vector saida, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = entrada[k] < 0.0 ? (entrada[k] * menor) : (entrada[k] * maior);
}


kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, REAL menor, REAL maior, int k0) {
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
kV softmaxExp(Vector entrada, Vector exponent,int k0) {
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
kV softmaxSomaExp(Vector eps, Vector soma, int saidatx, int saidaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL sum = 0;
	for (x = 0; x < saidatx; x++)
		for (y = 0; y < saidaty; y++) {
			sum += eps[KTensorMap(x, y, z, saidatx, saidaty)];
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
kV softmaxNormaliza(Vector exponet, Vector soma, Vector saida,int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)
	saida[k] = exponet[k] / soma[z];
}
/**
 * @goal Calcular os gradientes de entrada
 * @iteration dimensão da entrada a(x,y,z)
 * @param gradentrada Tensor de gradientes de entrada (escrita)
 * @param entrada Tensor de entrada (leitura)
 * @param gradnext Tensor gradiente da saída (leitura)
 * @param k0 usado internamente no kernel
 */
kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {
	int k = get_global_id(0) + k0;
	REAL xi = entrada[k];
	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];
}


#endif //GAB_KERNELS_OPENCL_H
