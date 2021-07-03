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

