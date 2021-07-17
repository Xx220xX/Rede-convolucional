kV convSum(Vector filtro, Vector entrada, Vector saida,
		   int passox,int passoy,
		   int saidatx, int saidaty,
		   int entradatx, int entradaty,
		   int fx,int fy, int fz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	TensorRemap(k, x, y, filtrok, saidatx, saidaty)
	double sum = 0, f, v;
	for (int m = 0; m < fx; m++)
		for (int n = 0; n < fy; n++)
			for (int z = 0; z < fz; z++) {
				f = filtro[TensorMap4D(m, n, z, filtrok, fx, fy, fz)];
				v = entrada[TensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty)];
				sum += f * v;
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
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			soma += entrada[TensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty)]
					* ds[TensorMap(i, j, l, saida_tx, saida_ty)];
		}
	}
	double dw = soma + gradFiltro[k] * momento;
	double w = filtros[k];
	filtros[k] = w - hitLearn * (dw + w * weightDecay);
	gradFiltro[k] = dw;
}

kV convCalcGradIn(Vector filtro,Vector gradEntrada,Vector gradNext,
				  int fx,int fy,int fz,
				  int passox,int passoy,
				  int entradatx,int entradaty,
				  int saidatx,int saidaty,int saidatz,
				  int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, entradatx, entradaty)

	Range range_filtro;
	range_filtro.min.x = 0;
	if (x + fx <= entradatx) {
		range_filtro.min.x = x + fx - entradatx;
	}
	range_filtro.max.x = fx - 1;
	if (x - fx + 1 < 0) {
		range_filtro.max.x = x;
	}
	range_filtro.min.y = 0;
	if (y + fy <= entradaty) {
		range_filtro.min.y = y + fy - entradaty;
	}
	range_filtro.max.y = fy - 1;
	if (y - fy + 1 < 0) {
		range_filtro.max.y = y;
	}
	double somaErro = 0, pesoAplicado = 0;
	int i, j;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / passox;
		if (i * passox + m != x) continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / passoy;
			if (j * passoy + n != y) continue;
			for (int w = 0; w < saidatz; w++) {
				pesoAplicado = filtro[TensorMap4D(m, n, z, w, fx, fy, fz)];
				somaErro += pesoAplicado * gradNext[TensorMap(i, j, w, saidatx, saidaty)];
			}
		}
	}
	gradEntrada[k] = somaErro;
}

