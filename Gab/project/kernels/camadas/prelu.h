kV preluativa(Vr entrada, Vr saida, Vr A, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		v = v * A[k];
	}
	saida[k] = v;
}

kV prelucalcgrad(Vr gradentrada, Vr entrada, Vr gradnext, Vr A, Vr dA, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
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

kV preluonlyfix(Vr entrada, Vr gradnext, Vr A, Vr dA, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		dA[k] = gradnext[k] + momento * dA[k];
	} else {
		dA[k] = momento * dA[k];
	}
	A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);
}

kV prelucalcgradBatch(Vr gradentrada, Vr entrada, Vr gradnext, Vr A, Vr dA, long batchSize, int k0) {
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

kV preluonlyDABatch(Vr entrada, Vr gradnext, Vr A, Vr dA, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = entrada[k];
	if (v < 0) {
		dA[k] = gradnext[k] / batchSize + dA[k];
	} else {
		dA[k] = 1.0 / batchSize + dA[k];
	}
}