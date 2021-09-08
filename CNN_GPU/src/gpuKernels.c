#ifndef GAB_KERNELS_OPENCL_H
#define GAB_KERNELS_OPENCL_H
//utils.h
// Created by Xx220xX on 10/05/2020.

#define Vector __global double *

#define kV __kernel void

#define TensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))

#define TensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))

#define TensorRemap4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\
_y_ = total%ty      ;                                        \
_x_ = (total - _y_)%(ty*tx)/ty ;                             \
_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \
_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);


#define TensorRemap(total, _x_, _y_, _z_, tx, ty)\
_y_ = total % ty;\
_x_ = ((total - _y_) % (ty * tx)) / ty;\
_z_ = (k - _x_ * ty - _y_) / (tx * ty);

#define TensorRemap2D(total, x, y, ty)\
y = total % ty;\
x = total/ ty;

typedef struct {
	int x, y, z;
} Ponto3d;

typedef struct {
	Ponto3d min, max;
} Range;

//bathnorm.h

// achar a media
kV BatchNormMedia(Vector entrada, Vector media,
                  int entradatx, int entradaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	double m = 0;
	for (x = 0; x < entradatx; x++) {
		for (y = 0; y < entradaty; y++) {
			m += entrada[TensorMap(x, y, z, entradatx, entradaty)];
		}
	}
	media[z] = m / (double) (entradatx * entradaty);
}

// achar a diferenca
kV BatchNormDiferenca(Vector entrada, Vector media,
                      Vector diferenca,
                      Vector diferencaquad,
                      int entradatx, int entradaty, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	diferenca[k] = entrada[k] - media[z];
	diferencaquad[k] = diferenca[k] * diferenca[k];
}

kV BatchNormVariance(Vector dif, Vector difQuad,
                     Vector sumdiferenca, Vector variancia,
                     double episolon, int diftx, int difty,
                     int k0) {
	int z = get_global_id(0) + k0;
	double sum = 0;
	double sumdif = 0;
	for (int x = 0; x < diftx; x++) {
		for (int y = 0; y < difty; y++) {
			sum += difQuad[TensorMap(x, y, z, diftx, difty)];
			sumdif += dif[TensorMap(x, y, z, diftx, difty)];
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
	TensorRemap(k, x, y, z, diferencatx, diferencaty)
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
	TensorRemap(k, x, y, z, entradatx, entradaty)
	double M = entradatx * entradaty;
	double dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);
	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);

	double didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;
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
	double sumY = 0;
	double sumB = 0;
	int k;
	for (int x = 0; x < entradatx; ++x) {
		for (int y = 0; y < entradaty; ++y) {
			k = TensorMap(x, y, z, entradatx, entradaty);
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
                        double hitlearn,
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
	TensorRemap(k, i, j, z, vx, vy)
	imi = imi + i;
	int imj = j + z * vy + z;
	out[imi * imy + imj] = ((int) v[k]) & 0xff;
}


kV
normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,
				int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}


kV subKernel(Vector grad, Vector saida, Vector target, int k0) {
	int k = get_global_id(0) + k0;
	grad[k] = saida[k] - target[k];
}

kV divKernel(Vector v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = v[k] / value;
}

kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = ((double) src[k]) / value;
}

kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {
//	int k = get_global_id(0) + k0;
//
//	int d;
//	for (int j = 0; j < noptiobs; j++) {
//		d = TensorMap4D(0, j, 0, k, 1, noptiobs, 1);
//		v[d] = (double) (j == ints[k]);
//	}
	int k = get_global_id(0) + k0;
	int x,y,z,w;
	TensorRemap4D(k,x,y,z,w,1,noptiobs,1);
	v[k] = (double) (y == ints[w]);
}




//conv.h
kV convSum(Vector filtro, Vector entrada, Vector saida,
           int passox, int passoy,
           int saidatx, int saidaty,
           int entradatx, int entradaty,
           int fx, int fy, int fz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	TensorRemap(k, x, y, filtrok, saidatx, saidaty)
	double sum = 0, f = 0, v = 0;
	int lf = 0, le = 0;
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			for (int z = 0; z < fz; z++) {
				lf = TensorMap4D(m, n, z, filtrok, fx, fy, fz);
				le = TensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);
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
                            double hitLearn, double momento, double weightDecay,
                            int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	TensorRemap4D(k, m, n, z, l, fx, fy, fz)
	double soma = 0;
	int le, ls;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			le = TensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);
			ls = TensorMap(i, j, l, saida_tx, saida_ty);
			soma += entrada[le]
			        * ds[ls];
		}
	}
	double dw = soma + gradFiltro[k] * momento;
	double w = filtros[k];
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
	TensorRemap(k, x, y, z, entradatx, entradaty)

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
	double somaErro = 0, pesoAplicado = 0;
	int i, j;
	int lf, ls;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / passox;
		if (i * passox + m != x) continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / passoy;
			if (j * passoy + n != y) continue;
			for (int w = 0; w < saidatz; w++) {
				lf = TensorMap4D(m, n, z, w, fx, fy, fz);
				ls = TensorMap(i, j, w, saidatx, saidaty);
				pesoAplicado = filtro[lf];
				somaErro += pesoAplicado * gradNext[ls];
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
	TensorRemap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d mapeado = {x * passox, y * passoy, 0};
	double sum = 0, f, v;
	for (int i = 0; i < fx; i++)
		for (int j = 0; j < fy; j++)
			for (int z = 0; z < entradatz; z++) {
				f = filtro[TensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];
				v = entrada[TensorMap(mapeado.x + i * largx, mapeado.y + j * largy, z, entradatx, entradaty)];

				sum += f * v;
			}
	saida[k] = sum;
}

kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,
				   double hitlearn,
                   double momento, double weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	double m = grad[k] + gradOld[k] * momento;
	double w = filtro[k];
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
	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)
	double soma = 0,aux;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			aux = entrada[TensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]
			        * ds[TensorMap(i, j, l, saida_tx, saida_ty)];
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
	TensorRemap(k, x, y, z, entradatx, entradaty)
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
	double somaErro = 0,aux, pesoAplicado = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		sx = (x - m * largx) / passox;
		if (sx * passox + m * largx != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			sy = (y - n * largy) / passox;
			if (sy * passoy + n * largy != y)continue;
			for (int l = 0; l < fz; l++) {
				pesoAplicado = filtro[TensorMap4D(m, n, z, l, fx, fy, fz)];
				aux = pesoAplicado * gradNext[TensorMap(sx, sy, l, saidatx, saidaty)];
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

double randomD(unsigned long seed,unsigned long id) {
	return (double) randoml(seed, id) / (double) MAX_INT_DP;
}

kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,
			 double pativa, int k0) {
	int i = get_global_id(0) + k0;
//	printf("kernel %lf %lf %g %g\n",randomD(seed, i),pativa,(double)(seed +i),(double)MAX_INT_DP);
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i];
}


kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i];
}

//fullconnect.h
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

double difsigmoid(double x) {
	double tmp = sigmoid(x);
	return tmp * (1.0 - tmp);
}

double tanghG(double x) { return tanh(x); }

double diftanhG(double x) {
	double tmp = tanh(x);
	return (1.0 - tmp * tmp);
}

double relu(double x) { return x > 0 ? x : 0.0; }

double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }

double func(int id, double x) {
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
		default:
			return 0;
	}
}

kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,
			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	double valorEntrada = 0;
	int n;
	for (n = 0; n < pesosy; n++) {
		valorEntrada += entrada[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];
	}
	z[m] = valorEntrada;
	saida[m] = func(funcaoativacao, valorEntrada);
}

kV fullfixweight(Vector a,
			  Vector pesos,
			  Vector dw,
			  Vector dz,
			  double hitlearn,
			  double decaimentoDePeso,
			  double momento,
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
	double soma = 0;
	for (int n = 0; n < pesosx; ++n) {
		soma += dz[n] * pesos[TensorMap(n, m, 0, pesosx, pesosy)];
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
	TensorRemap(k, x, y, z, txi, tyi)
	int s = TensorMap(x+t,y+l,z,txo,tyo);
	out[s] = in[k];
}
kV paddingBack(Vector gradNext,Vector gradin,
			   int txi,int tyi,
			   int txo,int tyo,
			   int t, int l , int k0){
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, txi, tyi)
	int s = TensorMap(x+t,y+l,z,txo,tyo);
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
	TensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	double mval, v;
	mval = -DBL_MAX;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
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
	TensorRemap(k, x, y, z, entradatx, entradaty)
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
	gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] =0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y)continue;
			if (entrada[TensorMap(x, y, z, entradatx, entradaty)] ==
				saida[TensorMap(i, j, z, saidatx, saidaty)]) {
				gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] =
						gradNext[TensorMap(i, j, z, saidatx, saidaty)];
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
	TensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	double soma = 0, v;

	for (int i = 0; i < fx; ++i) {
		for (int j = 0; j < fy; ++j) {
			soma += entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
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
	TensorRemap(k, x, y, z, entradatx, entradaty)
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
	double soma = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y)continue;
			soma += gradNext[TensorMap(i, j, z, saidatx, saidaty)];
		}
	}
	gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = soma / (fx * fy);

}


//relu.h
kV reluativa(Vector entrada, Vector saida, int k0) {
	int k = get_global_id(0) + k0;
	double v = entrada[k];
	if (v < 0)
		v = 0;
	saida[k] = v;
}

kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {
	int k = get_global_id(0) + k0;
	gradentrada[k] = entrada[k] <= 0.0 ? (0) : gradnext[k];
}

//softmax.h
kV SoftMaxativa1(Vector entrada, Vector exponent, Vector soma, int entradatx,
                 int entradaty,
                 int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	exponent[k] = exp(entrada[k]);
	soma[z] += exponent[k];
}

kV SoftMaxativa2(Vector exponet, Vector soma, Vector saida,
                 int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, saidatx, saidaty)
	saida[k] = exponet[TensorMap(x, y, z, saidatx, saidaty)] / soma[z];
}

kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {
	int k = get_global_id(0) + k0;
	double xi = entrada[k];
	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];
}


#endif //GAB_KERNELS_OPENCL_H
