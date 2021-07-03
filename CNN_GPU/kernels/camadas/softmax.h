kV SoftMaxativa1(Vector entrada, Vector exponent, Vector soma, int entradatx,
                 int entradaty,
                 int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	exponent[k] = exp(entrada[k]);
	soma[z] += exponent[k];
}

kV SoftMaxativa2(Vector exponet, Vector soma, Vector saida,
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

