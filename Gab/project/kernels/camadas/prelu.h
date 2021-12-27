kV preluativa(Vector entrada, Vector saida, Vector A, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		v = v * A[k];
	}
	saida[k] = v;
}

kV prelucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		gradentrada[k] = gradnext[k] * A[k];
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		gradentrada[k] = gradnext[k];
		dA[k] = momento * dA[k];
	}
	if (learn) {
		A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
	}
}

kV preluonlyfix(Vector entrada, Vector gradnext, Vector A, Vector dA, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		dA[k] = momento * dA[k];
	}
	A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
}

kV prelucalcgradBatch(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		gradentrada[k] = gradnext[k] * A[k];
		dA[k] = gradnext[k] / batchSize + dA[k];
	} else {
		gradentrada[k] = gradnext[k];
		dA[k] = 1.0 / batchSize + dA[k];
	}
}

kV preluonlyDABatch(Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		dA[k] = gradnext[k] / batchSize + dA[k];
	} else {
		dA[k] = 1.0 / batchSize + dA[k];
	}
}