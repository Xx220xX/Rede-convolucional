#define MAX_INT_DP  ((1UL << 31) - 1)
long randoml(unsigned long seed,unsigned long id) {
	seed += id;
	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;
}

double randomD(unsigned long seed,unsigned long id) {
	return (double) randoml(seed, id) / (double) MAX_INT_DP;
}

kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,
			 double pativa, int k0) {
	int i = get_global_id(0) + k0;
//	printf("kernel %lf %lf %g %g\n",randomD(seed, i),pativa,(double)(seed +i),(double)MAX_INT_DP);
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i];
}


kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i];
}
