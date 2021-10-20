kV preluativa(Vector entrada, Vector saida, Vector A, int k0) {
	int k = get_global_id(0) + k0;
	double v = entrada[k];
	if (v < 0)
		v = v * A[k];
	saida[k] = v;
}

kV prelucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA,
				 int learn,double hitlearn, double momento,
				 double decaimento,
				 int k0) {
	int k = get_global_id(0) + k0;
	double v = entrada[k];
	if (v < 0) {
		gradentrada[k] = gradnext[k] * A[k];
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		gradentrada[k] = gradnext[k];
		dA[k] = momento * dA[k];
	}
	if (learn)
		A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
}
