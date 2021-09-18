kV SoftMaxativa1(Vector entrada, Vector exponent,
				 int k0) {
	int k = get_global_id(0) + k0;
	exponent[k] = exp(entrada[k]);
}



kV SoftMaxativa2(Vector exponent, Vector soma,
				 int saidatx, int saidaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	int d;
	double sum;
	for (x = 0; x < saidatx; x++)
		for (y = 0; y < saidaty; y++) {
			d = TensorMap(x, y, z, saidatx, saidaty);
			sum += exponent[d];
		}
	soma[z] = sum;
}

kV SoftMaxativa3(Vector exponet, Vector soma, Vector saida,
				 int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, saidatx, saidaty)
	saida[k] = exponet[TensorMap(x, y, z, saidatx, saidaty)] / soma[z];
}
kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {
	int k = get_global_id(0) + k0;
	double xi = entrada[k];
	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];
}

