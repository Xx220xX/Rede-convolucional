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

kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j, z;
	TensorRemap(k, i, j, z, vx, vy)
	imi = imi + i;
	int imj = j + z * vy + z;
	out[imi * imy + imj] = ((int) v[k]) & 0xff;
}

kV printTensor(Vector t, int mx, int my, int mz, int ofset) {
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

kV norm(Vector v, Vector out, int len) {
	double s = 0,aux;
	for (int i = 0; i < len; ++i) {
		aux = v[i] * v[i];
		aux = (!(isnan(aux) || isinf(aux)))*aux;
		s += aux;
	}
	out[0] = pow(s, 0.5);
}

kV maxID(Vector v, Vector out, int len) {
	int s = 0;
	for (int i = 1; i < len; ++i) {
		if (v[s] < v[i]) {
			s = i;
		}
	}
	out[0] = (double) s;
}

kV
normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,
                int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}

kV findExtremes(Vector input, Vector output, int len) {
	double mn = input[0], mx = input[0];
	for (int i = 1; i < len; ++i) {
		if (input[i] > mx) mx = input[i];
		if (input[i] < mn) mn = input[i];
	}
	output[0] = mn;
	output[1] = mx;
}

kV sub(Vector grad, Vector saida, Vector target, int k0) {
	int k = get_global_id(0) + k0;
	grad[k] = saida[k] - target[k];
}

kV div(Vector v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = v[k] / value;
}

kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = ((double) src[k]) / value;
}

kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {
	int k = get_global_id(0) + k0;
	for (int j = 0; j < noptiobs; j++) {
		v[k * noptiobs + j] = (double) (j == ints[k]);
	}
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
//conv.h
//#include"utils.h"
kV convSum(Vector filtro, Vector entrada, Vector saida,
           int passo, int saidatx, int saidaty, int entradatx, int entradaty,
           int lenFilter, int entradatz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	TensorRemap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d mapeado = {x * passo, y * passo, 0};
	double sum = 0, f, v;
	for (int m = 0; m < lenFilter; m++)
		for (int n = 0; n < lenFilter; n++)
			for (int z = 0; z < entradatz; z++) {
				f = filtro[TensorMap4D(m, n, z, filtrok, lenFilter, lenFilter, entradatz)];
				v = entrada[TensorMap(mapeado.x + m, mapeado.y + n, z, entradatx, entradaty)];
				sum += f * v;
			}
	saida[k] = sum;

}

kV convFixWeight(Vector filtro, Vector grad, Vector gradOld, double hitlearn,
                 double momento, double weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	double m = grad[k] + gradOld[k] * momento;
	double w = filtro[k];
	filtro[k] = w - hitlearn * (m + w * weightDecay);
	gradOld[k] = m;
}

kV convCalcFiltro(     Vector ds,
					   Vector entrada,
					   Vector gradFiltro,
                       int gradFiltro_tx,
                       int gradFiltro_ty,
                       int gradFiltro_tz,
                       int entrada_tx,
                       int entrada_ty,
                       int saida_tx,
                       int saida_ty,
                       int passo,
                       int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
//	printf("kernel %d\n",k);
	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)
	double soma = 0,aux;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			aux = entrada[TensorMap(i*passo+m, j*passo+n,z,entrada_tx,entrada_ty)]
				   *ds[TensorMap(i,j,l,saida_tx,saida_ty)];
			aux = (!(isnan(aux) || isinf(aux)))*aux;
			soma +=aux;
		}
	}
	gradFiltro[k] = soma;
}

kV convCalcGrads(Vector filtro,
				 Vector entrada,
                 Vector gradEntrada,
                 Vector gradNext,
                 int lenFilter,
                 int filtroz,
                 int passo,
                 int entradatx,
                 int entradaty,
                 int saidatx,
                 int saidaty,
                 int numFilters,
                 int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	Range range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, numFilters);
	int minX, minY;
	double somaErro = 0, pesoAplicado = 0;
	double aux;
	for (int i = range.min.x; i <= range.max.x; i++) {
		minX = i * passo;
		for (int j = range.min.y; j <= range.max.y; j++) {
			minY = j * passo;
			for (int l = range.min.z; l <= range.max.z; l++) {
				pesoAplicado = filtro[TensorMap4D(x - minX, y - minY, z, l, lenFilter, lenFilter, filtroz)];
				aux = pesoAplicado * gradNext[TensorMap(i, j, l, saidatx, saidaty)];
				aux = (!(isnan(aux) || isinf(aux)))*aux;
				somaErro +=aux;
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
			aux = (!(isnan(aux) || isinf(aux)))*aux;
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
				aux = (!(isnan(aux) || isinf(aux)))*aux;
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

kV
fullfixweight(Vector a,
              Vector pesos,
              Vector dz,
              Vector dz_old,
              double hitlearn,
              double decaimentoDePeso,
              double momento,
              int inx,
              int iny,
              int inz,
              int pesosx,
              int pesosy,
              int k0) {
	int n = get_global_id(0) + k0;
	int m;
	double w;
	double tmp = dz[n] + dz_old[n] * momento;
	dz_old[n] = tmp;
	int k;
	for (m = inx * iny * inz - 1; m >= 0; m--) {
		k = TensorMap(n, m, 0, pesosx, pesosy);
		w = pesos[k];
		w -= hitlearn * (tmp * a[m] + w * decaimentoDePeso);
		pesos[k] = w;
	}
}

kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {
	int m = get_global_id(0) + k0;
	double aux = ds[m] * func(dfa, z[m]);
	aux = (!(isnan(aux) || isinf(aux)))*aux;
	dz[m] = aux;
}

kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,
                  int k0) {
	int m = get_global_id(0) + k0;
	double soma = 0,aux;
	for (int n = 0; n < pesosx; ++n) {
		aux = dz[n] * pesos[TensorMap(n, m, 0, pesosx, pesosy)];
		aux = (!(isnan(aux) || isinf(aux)))*aux;
		soma += aux;
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
kV poolativa(Vector entrada, Vector saida, int lenFilter,
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


kV
poolCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,
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


//poolav.h
kV PoolAvativa(Vector entrada, Vector saida, int lenFilter,
               int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passo, y * passo, 0};
	double soma = 0, v;

	for (int i = 0; i < lenFilter; ++i) {
		for (int j = 0; j < lenFilter; ++j) {
			soma += entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];

		}
	}
	saida[k] = soma / (lenFilter * lenFilter);
}


kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,
                   int lenFilter, int passo, int entradatx, int entradaty, int entradatz, int saidatx, int saidaty,
                   int k0) {
	int k = get_global_id(0) + k0;
	int x, y;
	TensorRemap2D(k, x, y, entradaty)
	double somaErro = 0;
	Range range;
	range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, 1);
	for (int z = 0; z < entradatz; ++z) {
		somaErro = 0;
		for (int i = range.min.x; i <= range.max.x; i++) {
			for (int j = range.min.y; j <= range.max.y; j++) {
				somaErro += gradNext[TensorMap(i, j, z, saidatx, saidaty)];
			}
		}
		gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = somaErro/ (lenFilter * lenFilter);
	}
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


