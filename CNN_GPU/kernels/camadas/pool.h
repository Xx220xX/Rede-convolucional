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

