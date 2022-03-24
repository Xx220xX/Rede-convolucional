kV poolativa(Vr entrada, Vw saida,Vw hmap, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;

	mval = entrada[kMap(mapeado.x , mapeado.y , z, entradatx, entradaty)];
	int mx = 0;
	int index;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			index = kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty);
			v = entrada[index];
			if (v > mval) {
				mval = v;
				mx = index;
			}
		}
	}
	saida[k] = mval;
	hmap[k] = mx;
}


kV poolCalcGrads(Vr A, Vr dA, Vr dS,  __global int * hmap, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, entradatx, entradaty)
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
	REAL soma = 0;

	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x) {
			continue;
		}
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y) {
				continue;
			}
			if (k == hmap[kMap(i, j, z, saidatx, saidaty)]) {
				soma += dS[kMap(i, j, z, saidatx, saidaty)];
			}
		}
	}
	dA[k] = soma;
}

