kV fullfeed(Vector a, Vector w, Vector b, Vector z, Vector s, int fid, int w_x, int w_y, int k0) {
	int m = get_global_id(0) + k0;
	REAL sum = 0;
	int n;
	for (n = 0; n < w_y; n++) {
		sum += a[n] * w[KTensorMap(m, n, 0, w_x, w_y)];
	}
	z[m] = sum + b[m];
	s[m] = func(fid, z[m]);
}

kV fullCalcDWandFix(Vector a, Vector w, Vector dw, Vector dz, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int pesosy, int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] + dw[k] * momento;
	w[k] = w[k] - hitlearn * (dw[k] + w[k] * decaimentoDePeso);
}


kV fullCalcDz(Vector dz, Vector ds, Vector z,int dfa, int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
}
kV fullCalcDzBath(Vector dz, Vector ds, Vector z,  Vector db, int dfa,long batchSize,  int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
	db[m] = dz[m]/batchSize + db[m];
}

kV fullCalcDzAndFixB(Vector dz, Vector ds, Vector z, Vector b, Vector db, int dfa, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0) {
	int m = get_global_id(0) + k0;
	dz[m] = ds[m] * func(dfa, z[m]);
	db[m] = dz[m] + db[m] * momento;
	b[m] = b[m] - hitlearn * (db[m] + b[m] * decaimentoDePeso);
}


kV fullcalcin(Vector dz, Vector da, Vector w, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	REAL soma = 0;
	for (int n = 0; n < pesosx; ++n) {
		soma += dz[n] * w[KTensorMap(n, m, 0, pesosx, pesosy)];
	}
	da[m] = soma;
}


kV fullCalcDWBatch(Vector a, Vector dw, Vector dz, long batchSize, int pesosy, int k0) {
	int k = get_global_id(0) + k0;
	int m, n;
	m = k / pesosy;
	n = k % pesosy;
	dw[k] = dz[m] * a[n] / batchSize + dw[k];
}

