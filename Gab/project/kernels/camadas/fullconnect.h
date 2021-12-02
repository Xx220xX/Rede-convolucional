
/**
 *
 * @param entrada (N,1,1)
 * @param pesos (M,N,1)
 * @param b (N,1,1)
 * @param z (N,1,1)
 * @param saida (N,1,1)
 * @param funcaoativacao
 * @param pesosx M
 * @param pesosy N
 * @param k0
 */
kV fullfeed(Vector entrada, Vector pesos, Vector b, Vector z, Vector saida,
			int funcaoativacao, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	REAL valorEntrada = 0;
	int n;
	for (n = 0; n < pesosy; n++) {
		valorEntrada += entrada[n] * pesos[KTensorMap(m, n, 0, pesosx, pesosy)];
	}
	z[m] = valorEntrada + b[m];
	saida[m] = func(funcaoativacao, z[m]);
}

kV fullCalcDWandFix(Vector a,
					Vector w,
					Vector dw,
					Vector dz,
					REAL hitlearn,
					REAL momento,
					REAL decaimentoDePeso,
					int pesosy,
					int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] + dw[k] * momento;
	w[k] = w[k] - hitlearn * (dw[k] + w[k] * decaimentoDePeso);
}

kV fullCalcDz(Vector dz, Vector ds, Vector z, Vector b, Vector db,
			  int dfa, REAL hitlearn,
			  REAL momento, REAL decaimentoDePeso,
			  int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
}

kV fullCalcDzAndFixB(Vector dz, Vector ds, Vector z, Vector b,
					 Vector db, int dfa, REAL hitlearn,
					 REAL momento, REAL decaimentoDePeso,
					 int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
	db[m] = dz[m] + db[m] * momento;
	b[m] = b[m] - hitlearn * (db[m] + b[m] * decaimentoDePeso);
}

kV fullcalcin(Vector dz, Vector da, Vector w, int pesosx, int pesosy,
			  int k0) {
	int m = get_global_id(0) + k0;
	REAL soma = 0;
	for (int n = 0; n < pesosx; ++n) {
		soma += dz[n] * w[KTensorMap(n, m, 0, pesosx, pesosy)];
	}
	da[m] = soma;
}
