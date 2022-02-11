#define MAX_INT_DP  ((1UL << 31) - 1)

long randoml(unsigned long seed, unsigned long id) {
	seed += id;
	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;
}

REAL randomD(unsigned long seed, unsigned long id) {
	return (REAL) randoml(seed, id) / (REAL) MAX_INT_DP;
}

kV dropativa(Vr entrada, Vr saida, __global char *hitmap, long seed, REAL pativa, int k0) {
	int i = get_global_id(0) + k0;
//	printf("kernel %lf %lf %g %g\n",randomD(seed, i),pativa,(REAL)(seed +i),(REAL)MAX_INT_DP);
	char teste = (char) (randomD(seed, i) <= pativa);
	hitmap[i] = teste;
	saida[i] = teste * entrada[i]/pativa;
}


kV dropcalcgrad(Vr gradentrada, __global char *hitmap, Vr gradnext,REAL pativa, int k0) {
	int i = get_global_id(0) + k0;
	gradentrada[i] = hitmap[i] * gradnext[i]/pativa;
}
