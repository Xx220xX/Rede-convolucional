kV poolAVativa(Vector entrada, Vector saida,
			   int passox, int passoy,
			   int fx, int fy,
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

