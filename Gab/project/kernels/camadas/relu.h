
kV reluativa(Vr A, Vr S, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	S[k] = A[k] < 0.0 ? (A[k] * menor) : (A[k] * maior);
}


kV relucalcgrad(Vr dA, Vr A, Vr dS, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	dA[k] = A[k] < 0.0 ? (menor * dS[k]) : (maior * dS[k]);
}
