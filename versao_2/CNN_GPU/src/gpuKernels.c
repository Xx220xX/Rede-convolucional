#ifndef GAB_KERNELS_OPENCL_H
#define GAB_KERNELS_OPENCL_H
//utils.h
// Created by Xx220xX on 10/05/2020.
#ifndef ATIVATIONSFUNCTIONS_H
#define ATIVATIONSFUNCTIONS_H
#define  REAL float

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
	return 1.0 / (1.0 + exp(-x));
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

REAL func(int id, REAL x) {
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
		default:
			return 0;
	}
}

#endif
//bathnorm.h

// achar a media
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
	media[z] = m / (REAL) (entradatx * entradaty);
}

// achar a diferenca
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
	variancia[z] = sqrt(sum / (difty * diftx) + episolon);
}

// normaliza
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
                       Vector gradY,
                       Vector gradB,
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
	gradB[z] = sumB;
	gradY[z] = sumY;
}


kV batchNormCorrigePeso(Vector gradY,
                        Vector gradB,
                        Vector Y,
                        Vector B,
                        REAL hitlearn,
                        int k0) {
	int z = get_global_id(0) + k0;
	B[z] = B[z] - gradB[z] * hitlearn;
	Y[z] = Y[z] - gradY[z] * hitlearn;
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
	imagem_saida[(i+i0)*width+j+j0] = ((int) v[KTensorMap(x, y, z, vx, vy)] ) & 0xff;
}


kV normalizeVector(Vector input, Vector saida, REAL multiplicador, REAL somador, REAL subtrator,
				   int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}


kV subKernel(Vector grad, Vector saida, Vector target, int k0) {
	int k = get_global_id(0) + k0;
	grad[k] = saida[k] - target[k];
}

kV divKernel(Vector v, REAL value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = v[k] / value;
}

kV divIntDo(__global unsigned char *src, Vector v, REAL value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = ((REAL) src[k]) / value;

}

kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {
	int w = get_global_id(0) + k0;
	int y = ints[w];
	v[KTensorMap4D(0, y, 0, w, 1, noptiobs, 1)] = 1.0;
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
	REAL sum = 0, f , v ;
	int lf, le ;
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
	saida[k] = func(fid,sum);
}

kV convFCalcGradZ(Vector  ds,Vector z,Vector dz,int fid,int k0){
	int k = get_global_id(0) + k0;
	dz[k] = ds[k]*func(fid,z[k]);
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


//convNc.h
//#include"utils.h"
kV convncSum(Vector filtro, Vector entrada, Vector saida,
             int passox, int passoy, int largx,
             int largy, int saidatx, int saidaty,
             int entradatx, int entradaty,int fx, int fy,
             int entradatz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	KTensorRemap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d Kmapeado = {x * passox, y * passoy, 0};
	REAL sum = 0, f, v;
	for (int i = 0; i < fx; i++)
		for (int j = 0; j < fy; j++)
			for (int z = 0; z < entradatz; z++) {
				f = filtro[KTensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];
				v = entrada[KTensorMap(Kmapeado.x + i * largx, Kmapeado.y + j * largy, z, entradatx, entradaty)];

				sum += f * v;
			}
	saida[k] = sum;
}

kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,
				   REAL hitlearn,
                   REAL momento, REAL weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	REAL m = grad[k] + gradOld[k] * momento;
	REAL w = filtro[k];
	filtro[k] = w - hitlearn * (m + w * weightDecay);
	gradOld[k] = m;
}

kV convncCalcFiltro(Vector ds,
                    Vector entrada,
                    Vector gradFiltro,
                    int gradFiltro_tx,
                    int gradFiltro_ty,
                    int gradFiltro_tz,

                    int entrada_tx,
                    int entrada_ty,

                    int saida_tx,
                    int saida_ty,

                    int passox,
                    int passoy,

                    int largx,
                    int largy,
                    int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	KTensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)
	REAL soma = 0,aux;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			aux = entrada[KTensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]
			        * ds[KTensorMap(i, j, l, saida_tx, saida_ty)];
			//aux = (!(isnan(aux) || isinf(aux)))*aux;
			soma += aux;
		}
	}
	gradFiltro[k] = soma;
}

/**
 * equacao a ser implementada
 * x = s*p + m*w
 * onde:
 * 	x é da entrada 
 * 	s é da saida
 * 	m é do filtro
 * 	s = (x - m*w)/p
 */
kV convncCalcGrads(Vector filtro,
                   Vector entrada,
                   Vector gradEntrada,
                   Vector gradNext,

                   int passox,
                   int passoy,
                   int largx,
                   int largy,

                   int entradatx,
                   int entradaty,
                   int saidatx,
                   int saidaty,

                   int fx,
                   int fy,
                   int fz,
                   int numFilters,

                   int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, entradatx, entradaty)
	Range range_filtro ;
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
	REAL somaErro = 0,aux, pesoAplicado = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		sx = (x - m * largx) / passox;
		if (sx * passox + m * largx != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			sy = (y - n * largy) / passox;
			if (sy * passoy + n * largy != y)continue;
			for (int l = 0; l < fz; l++) {
				pesoAplicado = filtro[KTensorMap4D(m, n, z, l, fx, fy, fz)];
				aux = pesoAplicado * gradNext[KTensorMap(sx, sy, l, saidatx, saidaty)];
				//aux = (!(isnan(aux) || isinf(aux)))*aux;
				somaErro +=aux;
			}
		}
	}
	gradEntrada[k] = somaErro;
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


kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,
			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	REAL valorEntrada = 0;
	int n;
	for (n = 0; n < pesosy; n++) {
		valorEntrada += entrada[n] * pesos[KTensorMap(m, n, 0, pesosx, pesosy)];
	}
	z[m] = valorEntrada;
	saida[m] = func(funcaoativacao, valorEntrada);
}

kV fullfixweight(Vector a,
			  Vector pesos,
			  Vector dw,
			  Vector dz,
			  REAL hitlearn,
			  REAL decaimentoDePeso,
			  REAL momento,
			  int pesosy,
			  int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] + dw[k] * momento;
	pesos[k] = pesos[k] - hitlearn * (dw[k] + pesos[k] * decaimentoDePeso);
}

kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
}

kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,
				  int k0) {
	int m = get_global_id(0) + k0;
	REAL soma = 0;
	for (int n = 0; n < pesosx; ++n) {
		soma += dz[n] * pesos[KTensorMap(n, m, 0, pesosx, pesosy)];
	}
	da[m] = soma;
}

//padding.h
kV paddingfeed(Vector in,Vector out,
			   int txi,int tyi,
			   int txo,int tyo,
			   int t, int l ,
			   int k0){
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, txi, tyi)
	int s = KTensorMap(x+t,y+l,z,txo,tyo);
	out[s] = in[k];
}
kV paddingBack(Vector gradNext,Vector gradin,
			   int txi,int tyi,
			   int txo,int tyo,
			   int t, int l , int k0){
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, txi, tyi)
	int s = KTensorMap(x+t,y+l,z,txo,tyo);
	gradin[k] = gradNext[s];
}
//pool.h
kV poolativa(Vector entrada, Vector saida,
			 int passox,int passoy,
			 int filtrox,int filtroy,
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
	gradEntrada[KTensorMap(x, y, z, entradatx, entradaty)] =0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y)continue;
			if (entrada[KTensorMap(x, y, z, entradatx, entradaty)] ==
				saida[KTensorMap(i, j, z, saidatx, saidaty)]) {
				gradEntrada[KTensorMap(x, y, z, entradatx, entradaty)] =
						gradNext[KTensorMap(i, j, z, saidatx, saidaty)];
				return;
			}
		}
	}

}


//poolav.h
kV PoolAvativa(Vector entrada, Vector saida,
			   int passox,int passoy,
			   int fx,int fy,
			   int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL soma = 0, v;

	for (int i = 0; i < fx; ++i) {
		for (int j = 0; j < fy; ++j) {
			soma += entrada[KTensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
		}
	}
	saida[k] = soma / (fx * fy);
}


kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,
                   int px, int py,  int fx, int fy,
				   int entradatx, int entradaty, int saidatx, int saidaty,
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


//prelu.h
kV preluativa(Vector entrada, Vector saida, Vector A, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0)
		v = v * A[k];
	saida[k] = v;
}

kV prelucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA,
				 int learn,REAL hitlearn, REAL momento,
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

//relu.h
kV reluativa(Vector entrada, Vector saida, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = entrada[k] < 0.0 ? (entrada[k] * menor) : (entrada[k]* maior);
}

kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	gradentrada[k] = entrada[k] < 0.0 ? (menor*gradnext[k]) : (maior*gradnext[k]);
}

//softmax.h
kV SoftMaxativa1(Vector entrada, Vector exponent,
				 int k0) {
	int k = get_global_id(0) + k0;
	exponent[k] = exp(entrada[k]);
}



kV SoftMaxativa2(Vector exponent, Vector soma,
				 int saidatx, int saidaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	int d;
	REAL sum;
	for (x = 0; x < saidatx; x++)
		for (y = 0; y < saidaty; y++) {
			d = KTensorMap(x, y, z, saidatx, saidaty);
			sum += exponent[d];
		}
	soma[z] = sum;
}

kV SoftMaxativa3(Vector exponet, Vector soma, Vector saida,
				 int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)
	saida[k] = exponet[KTensorMap(x, y, z, saidatx, saidaty)] / soma[z];
}
kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {
	int k = get_global_id(0) + k0;
	REAL xi = entrada[k];
	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];
}


#endif //GAB_KERNELS_OPENCL_H
