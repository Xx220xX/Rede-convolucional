
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



