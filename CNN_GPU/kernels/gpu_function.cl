//
// Created by Xx220xX on 10/05/2020.
//
#ifndef CL_KERNEL_SRC_H
#define CL_KERNEL_SRC_H

static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }

static double difsigmoid(double x) {
	double tmp = sigmoid(x);
	return tmp * (1 - tmp);
}

static double tanghG(double x) { return tanh(x); }

static double diftanhG(double x) {
	double tmp = tanh(x);
	return (1.0 - tmp * tmp);
}

static double relu(double x) { return x > 0 ? x : 0.0; }

static double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }

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

#define TensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))
#define TensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))


#define TensorRemap(total, x, y, z, tx, ty)\
y = total % ty;\
x = ((total - y) % (ty * tx)) / ty;\
z = (k - x * ty - y) / (tx * ty);

#define TensorRemap2D(total, x, y, ty)\
y = total % ty;\
x = total/ ty;


typedef struct {
	int x, y, z;
} Ponto3d;
typedef struct {
	Ponto3d min, max;
} Range;

__kernel void createImg(__global unsigned char *out,__global double *v,int vx,int vy,int imi, int imy,int k0){
	int k =  get_global_id(0) + k0;
	int i,j,z;
	TensorRemap(k, i,j, z, vx, vy)
	imi = imi+i;
	int imj = j+z*vy+z;
	out[imi*imy+imj]= ((int)v[k])&0xff;
}
__kernel void printTensor(__global double *t, int mx, int my, int mz, int ofset) {
	for (int z = 0; z < mz; z++) {
		printf("[Dim%d]\n", z);
		for (int x = 0; x < mx; x++) {
			for (int y = 0; y < my; y++) {

				printf("%.4lf \t", t[TensorMap(x, y, z, mx, my) + ofset]);
			}
			printf("\n");
		}
	}
}

__kernel void norm(__global double *v, __global double *out, int len) {
	double s = 0;
	for (int i = 0; i < len; ++i) {
		s += v[i] * v[i];
	}
	out[0] = pow(s, 0.5);
}

__kernel void maxID(__global double *v, __global double *out, int len) {
	int s = 0;
	for (int i = 1; i < len; ++i) {
		if (v[s] < v[i]) {
			s = i;
		}
	}
	out[0] = (double) s;
}

__kernel void
normalizeVector(__global double *input, __global double *saida, double multiplicador, double somador, double subtrator,
          int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}

__kernel void findExtremes(__global double *input, __global double *output, int len) {
	double mn = input[0], mx = input[0];
	for (int i = 1; i < len; ++i) {
		if(input[i]>mx) mx = input[i];
		if(input[i]<mn) mn = input[i];
	}
	output[0] = mn;
	output[1] = mx;
}

__kernel void sub(__global double *grad, __global double *saida, __global double *target, int k0) {
	int k = get_global_id(0) + k0;
	grad[k] = saida[k] - target[k];
}

__kernel void div(__global double *v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = v[k] / value;
}

__kernel void divIntDo(__global unsigned char *src, __global double *v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = ((double) src[k]) / value;
}

__kernel void int2vector(__global unsigned char *ints, __global double *v, int noptiobs, int k0) {
	int k = get_global_id(0) + k0;
	for (int j = 0; j < noptiobs; j++) {
		v[k * noptiobs + j] = (double) (j == ints[k]);
	}
}


__kernel void convSum(__global double *filtro, __global double *entrada, __global double *saida,
                      int passo, int saidatx, int saidaty, int entradatx, int entradaty,
                      int lenFilter, int entradatz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	TensorRemap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d mapeado = {x * passo, y * passo, 0};
	double sum = 0, f, v;
	for (int i = 0; i < lenFilter; i++)
		for (int j = 0; j < lenFilter; j++)
			for (int z = 0; z < entradatz; z++) {
				f = filtro[TensorMap4D(i, j, z, filtrok, lenFilter, lenFilter, entradatz)];
				v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
				sum += f * v;
			}
	saida[TensorMap(x, y, filtrok, saidatx, saidaty)] = sum;
}


int normaliza_range(double f, int max, int lim_min) {
	if (f <= 0)return 0;
	if (f >= max - 1)return max - 1;
	if (lim_min) return ceil(f);
	else return floor(f);
}

Range mapeia_entrada_saida(int x, int y, int passo, int tamanhoFiltro, int saidatx, int saidaty, int numeroFiltros) {
	double a = x, b = y;
	Range r;
	r.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saidatx, 1);
	r.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saidaty, 1);
	r.min.z = 0;

	r.max.x = normaliza_range(a / passo, saidatx, 0);
	r.max.y = normaliza_range(b / passo, saidaty, 0);
	r.max.z = numeroFiltros - 1;
	return r;
}

__kernel void convFixWeight(__global double *filtro, __global double *grad, __global double *gradOld, double hitlearn,
                            double momento, double multp, double weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	double m = grad[k] + gradOld[k] * momento;
	double w = filtro[k];
	filtro[k] = w - hitlearn * (m * multp + w * weightDecay);
	gradOld[k] = m;
}

__kernel void convCalcGrads(__global double *filtro, __global double *gradFiltro, __global double *entrada,
                            __global double *gradEntrada,
                            __global double *gradNext, int lenFilter, int filtroz, int passo, int entradatx,
                            int entradaty, int saidatx, int saidaty,
                            int numFilters, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, entradatx, entradaty)

	Range range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, numFilters);
	int minX, minY;
	double somaErro = 0, pesoAplicado = 0;
	for (int i = range.min.x; i <= range.max.x; i++) {
		minX = i * passo;
		for (int j = range.min.y; j <= range.max.y; j++) {
			minY = j * passo;
			for (int l = range.min.z; l <= range.max.z; l++) {
				pesoAplicado = filtro[TensorMap4D(x - minX, y - minY, z, l, lenFilter, lenFilter, filtroz)];
				somaErro += pesoAplicado * gradNext[TensorMap(i, j, l, saidatx, saidaty)];
				gradFiltro[TensorMap4D(x - minX, y - minY, z, l, lenFilter, lenFilter, filtroz)] +=
						entrada[k] * gradNext[TensorMap(i, j, l, saidatx, saidaty)];

			}
		}
	}
	gradEntrada[k] = somaErro;
}


__kernel void fullfeed(__global double *entrada, __global double *pesos, __global double *input, __global double *saida,
                       int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
	int n = get_global_id(0) + k0;
	double valorEntrada = 0;
	int m;
	for (int x = 0; x < inx; x++)
		for (int y = 0; y < iny; y++)
			for (int z = 0; z < inz; z++) {
				m = TensorMap(x, y, z, inx, iny);//z * (inx *iny) + y * inx + x;
				valorEntrada += entrada[m] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];
			}
	input[n] = valorEntrada;
	saida[n] = func(funcaoativacao, valorEntrada);
}

__kernel void
fullfixweight(__global double *entrada, __global double *pesos, __global double *grad, __global double *oldgrad,
              double hitlearn, double decaimentoDePeso, double momento,
              int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
	int n = get_global_id(0) + k0;
	int m;
	double w;
	double tmp = grad[n] + oldgrad[n] * momento;

	for (int i = 0; i < inx; ++i) {
		for (int j = 0; j < iny; ++j) {
			for (int z = 0; z < inz; ++z) {
				m = TensorMap(i, j, z, inx, iny);
				w = pesos[TensorMap(m, n, 0, pesosx, pesosy)];
				w -= hitlearn * (tmp * entrada[TensorMap(i, j, z, inx, iny)] + w * decaimentoDePeso);
				pesos[TensorMap(m, n, 0, pesosx, pesosy)] = w;
			}
		}
	}
	oldgrad[n] = tmp;
}

__kernel void
fullcalcgrads1(__global double *grad, __global double *gradNext, __global double *input, int dfa, int k0) {
	int n = get_global_id(0) + k0;
	grad[n] = gradNext[n] * func(dfa, input[n]);
}

__kernel void
fullcalcgrads2(__global double *grad, __global double *gradsEntrada, __global double *pesos, int pesosx, int pesosy,
               int k0) {
	int m = get_global_id(0) + k0;
	gradsEntrada[m] = 0;
	for (int n = 0; n < pesosy; ++n) {
		gradsEntrada[m] += grad[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];
	}
}


long randoml(long seed, long id) {
	seed += id;
	return (seed * 0x5deece66dL + 0xbL) & ((1L << 48) - 1);
}

double randomD(long seed, long id) {
	return (double) randoml(seed, id) / (double) ((1L << 48) - 1);
}

__kernel void
dropativa(__global double *entrada, __global double *saida, __global char *hitmap, long seed, double pativa, int k0) {
	int i = get_global_id(0) + k0;
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i];
}


__kernel void dropcalcgrad(__global double *gradentrada, __global char *hitmap, __global double *gradnext, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i];
}

//### guilherme


__kernel void reluativa(__global double *entrada, __global double *saida, int k0) {
	int k = get_global_id(0) + k0;
	double v = entrada[k];
	if (v < 0)
		v = 0;
	saida[k] = v;
}

__kernel void relucalcgrad(__global double *gradentrada, __global double *entrada, __global double *gradnext, int k0) {
	int k = get_global_id(0) + k0;
	gradentrada[k] = entrada[k] <= 0.0 ? (0) : gradnext[k];
}

__kernel void poolativa(__global double *entrada, __global double *saida, int lenFilter,
                        int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passo, y * passo, 0};
	double mval, v;
	mval = -DBL_MAX;
	for (int i = 0; i < lenFilter; ++i) {
		for (int j = 0; j < lenFilter; ++j) {
			v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
			if (v > mval)
				mval = v;
		}
	}
	saida[k] = mval;
}


__kernel void
poolCalcGrads(__global double *entrada, __global double *gradEntrada, __global double *gradNext, __global double *saida,
              int lenFilter, int passo, int entradatx, int entradaty, int entradatz, int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y;
	TensorRemap2D(k, x, y, entradaty)
	double somaErro = 0, testeMax;
	Range range;
	range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, 1);
	for (int z = 0; z < entradatz; ++z) {
		somaErro = 0;
		for (int i = range.min.x; i <= range.max.x; i++) {
			for (int j = range.min.y; j <= range.max.y; j++) {
				testeMax = (entrada[TensorMap(x, y, z, entradatx, entradaty)] ==
				            saida[TensorMap(i, j, z, saidatx, saidaty)]);
				somaErro += testeMax * gradNext[TensorMap(i, j, z, saidatx, saidaty)];
			}
		}
		gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = somaErro;
	}
}


#endif //CL_TESTE_KERNEL_SRC_H