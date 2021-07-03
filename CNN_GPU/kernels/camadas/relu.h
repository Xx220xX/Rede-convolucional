kV reluativa(Vector entrada, Vector saida, int k0) {
	int k = get_global_id(0) + k0;
	double v = entrada[k];
	if (v < 0)
		v = 0;
	saida[k] = v;
}

kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {
	int k = get_global_id(0) + k0;
	gradentrada[k] = entrada[k] <= 0.0 ? (0) : gradnext[k];
}
