long randoml(long seed, long id) {
	seed += id;
	return (seed * 0x5deece66dL + 0xbL) & ((1L << 48) - 1);
}

double randomD(long seed, long id) {
	return (double) randoml(seed, id) / (double) ((1L << 48) - 1);
}

kV
dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed, double pativa, int k0) {
	int i = get_global_id(0) + k0;
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i];
}


kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i];
}
