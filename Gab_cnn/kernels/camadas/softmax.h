/**
 * @goal calcular e^a(x,y,z)
 * @iteration dimensão de a (x,y,z)
 * @param entrada Tensor de entrada (leitura)
 * @param exponent Tensor e^entrada (escrita)
 * @param k0 usado internamente no kernel
 */
kV softmaxExp(Vector entrada, Vector exponent,int k0) {
	int k = get_global_id(0) + k0;
	exponent[k] = EXP(entrada[k]);
}


/***
 * @goal encontrar a soma de cada dimensão z
 * @iteration dimensão z da entrada a(:,:,z)
 * @param eps Tensor exponencial da entrada (leitura)
 * @param soma Tensor da soma das exponenciais (escrita)
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão x da saída
 * @param k0 usado internamente no kernel
 */
kV softmaxSomaExp(Vector eps, Vector soma, int saidatx, int saidaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL sum = 0;
	for (x = 0; x < saidatx; x++)
		for (y = 0; y < saidaty; y++) {
			sum += eps[KTensorMap(x, y, z, saidatx, saidaty)];
		}
	soma[z] = sum;
}
/***
 * @goal Normalizar a exponencial pela soma
 *  * @iteration dimensão da saída  s(x,y,z)
 * @param exponet Tensor exponencial da entrada (leitura)
 * @param soma Tensor da soma das exponenciais (leitura)
 * @param saida Tensor de saída (escrita)
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão x da saída
 * @param k0 usado internamente no kernel
 */
kV softmaxNormaliza(Vector exponet, Vector soma, Vector saida,int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)
	saida[k] = exponet[k] / soma[z];
}
/**
 * @goal Calcular os gradientes de entrada
 * @iteration dimensão da entrada a(x,y,z)
 * @param gradentrada Tensor de gradientes de entrada (escrita)
 * @param entrada Tensor de entrada (leitura)
 * @param gradnext Tensor gradiente da saída (leitura)
 * @param k0 usado internamente no kernel
 */
kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {
	int k = get_global_id(0) + k0;
	REAL xi = entrada[k];
	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];
}

