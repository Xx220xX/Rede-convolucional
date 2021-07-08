double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

double difsigmoid(double x) {
	double tmp = sigmoid(x);
	return tmp * (1.0 - tmp);
}

double tanghG(double x) { return tanh(x); }

double diftanhG(double x) {
	double tmp = tanh(x);
	return (1.0 - tmp * tmp);
}

double relu(double x) { return x > 0 ? x : 0.0; }

double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }

double func(int id, double x) {
	switch (id) {
		case 0:
			return sigmoid(x);
		case 1:
			return difsigmoid(x);
		case 2:
			return tanghG(x);
		case 3:
			return diftanhG(x);
		case 4:
			return relu(x);
		case 5:
			return difrelu(x);
		default:
			return 0;
	}
}

kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,
            int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
	int m = get_global_id(0) + k0;
	double valorEntrada = 0;
	int n;
	for (n = 0; n < pesosy; n++) {
		valorEntrada += entrada[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];
	}
	z[m] = valorEntrada;
	saida[m] = func(funcaoativacao, valorEntrada);
}

kV
fullfixweight(Vector a,
              Vector pesos,
              Vector dz,
              Vector dz_old,
              double hitlearn,
              double decaimentoDePeso,
              double momento,
              int inx,
              int iny,
              int inz,
              int pesosx,
              int pesosy,
              int k0) {
	int n = get_global_id(0) + k0;
	int m;
	double w;
	double tmp = dz[n] + dz_old[n] * momento;
	dz_old[n] = tmp;
	int k;
	for (m = inx * iny * inz - 1; m >= 0; m--) {
		k = TensorMap(n, m, 0, pesosx, pesosy);
		w = pesos[k];
		w -= hitlearn * (tmp * a[m] + w * decaimentoDePeso);
		pesos[k] = w;
	}
}

kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {
	int m = get_global_id(0) + k0;
	double aux = ds[m] * func(dfa, z[m]);
	aux = (!(isnan(aux) || isinf(aux)))*aux;
	dz[m] = aux;
}

kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,
                  int k0) {
	int m = get_global_id(0) + k0;
	double soma = 0,aux;
	for (int n = 0; n < pesosx; ++n) {
		aux = dz[n] * pesos[TensorMap(n, m, 0, pesosx, pesosy)];
		aux = (!(isnan(aux) || isinf(aux)))*aux;
		soma += aux;
	}
	da[m] = soma;
}
