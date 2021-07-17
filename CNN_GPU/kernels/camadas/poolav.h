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
				   int fx, int fy, int px, int py,
				   int entradatx, int entradaty, int saidatx, int saidaty,
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

