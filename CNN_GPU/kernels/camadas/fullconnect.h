

kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,
			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	REAL valorEntrada = 0;
	int n;
	for (n = 0; n < pesosy; n++) {
		valorEntrada += entrada[n] * pesos[KTensorMap(m, n, 0, pesosx, pesosy)];
	}
	z[m] = valorEntrada;
	saida[m] = func(funcaoativacao, valorEntrada);
}

kV fullfixweight(Vector a,
			  Vector pesos,
			  Vector dw,
			  Vector dz,
			  REAL hitlearn,
			  REAL decaimentoDePeso,
			  REAL momento,
			  int pesosy,
			  int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] + dw[k] * momento;
	pesos[k] = pesos[k] - hitlearn * (dw[k] + pesos[k] * decaimentoDePeso);
}

kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
}

kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,
				  int k0) {
	int m = get_global_id(0) + k0;
	REAL soma = 0;
	for (int n = 0; n < pesosx; ++n) {
		soma += dz[n] * pesos[KTensorMap(n, m, 0, pesosx, pesosy)];
	}
	da[m] = soma;
}
