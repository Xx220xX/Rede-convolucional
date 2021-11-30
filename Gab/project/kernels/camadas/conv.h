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

