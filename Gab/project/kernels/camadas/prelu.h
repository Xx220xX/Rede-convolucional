kV preluativa(Vr A, Vw S, Vr W, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		v = v * W[k];
	}
	S[k] = v;
}

kV prelucalcgrad(Vw dA, Vr A, Vr dS, Vrw W, Vrw dW, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dA[k] = dS[k] * W[k];
		dW[k] = dS[k] + momento * dW[k];
	} else {
		dA[k] = dS[k];
		dW[k] = momento * dW[k];
	}
	if (learn) {
		W[k] = W[k] - hitlearn * (dW[k] + W[k] * decaimento);
	}
}

kV preluonlyfix(Vr A, Vr dS, Vrw W, Vrw dW, REAL hitlearn, REAL momento, REAL decaimento, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dW[k] = dS[k] + momento * dW[k];
	} else {
		dW[k] = momento * dW[k];
	}
	W[k] = W[k] - hitlearn * (dW[k] + W[k] * decaimento);
}

kV prelucalcgradBatch(Vw dA, Vr A, Vr dS, Vr W, Vrw dW, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dA[k] = dS[k] * W[k];
		dW[k] = dS[k] / batchSize + dW[k];
	} else {
		dA[k] = dS[k];
		dW[k] = 1.0 / batchSize + dW[k];
	}
}

kV preluonlyDABatch(Vr A, Vr dS, Vr W, Vr dW, long batchSize, int k0) {
	int k = get_global_id(0) + k0;
	REAL v = A[k];
	if (v < 0) {
		dW[k] = dS[k] / batchSize + dW[k];
	} else {
		dW[k] = 1.0 / batchSize + dW[k];
	}
}