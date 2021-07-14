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





kV
normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,
                int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
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
