kV reluativa(Vector entrada, Vector saida, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = entrada[k] < 0.0 ? (entrada[k] * menor) : (entrada[k]* maior);
}

kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, REAL menor, REAL maior, int k0) {
	int k = get_global_id(0) + k0;
	gradentrada[k] = entrada[k] < 0.0 ? (menor*gradnext[k]) : (maior*gradnext[k]);
}
