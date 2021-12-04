/**
 * @goal calcular e^a(x,y,z)
 * @iteration dimensão de a (x,y,z)
 * @param entrada Tensor de entrada (leitura)
 * @param exponent Tensor e^entrada (escrita)
 * @param k0 usado internamente no kernel
 */
kV softmaxExp(Vector entrada, Vector exponent, int k0) {
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
	for (x = 0; x < saidatx; x++) {
		for (y = 0; y < saidaty; y++) {
			sum += eps[KTensorMap(x, y, z, saidatx, saidaty)];
		}
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
kV softmaxNormaliza(Vector exponet, Vector soma, Vector saida, int saidatx, int saidaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, saidatx, saidaty)
	saida[k] = exponet[k] / soma[z];
}
/**
 * @goal Calcular os gradientes de entrada
 * @iteration dimensão da entrada a(x,y,z)
 * @param da Tensor de gradientes de entrada (escrita)
 * @param s Tensor de saida (leitura)
 * @param ds Tensor gradiente da saída (leitura)
 * @param sx dimensão x da saída
 * @param sy dimensão y da saída
 * @param k0 usado internamente no kernel
 */
kV softMaxcalcgrad(Vector da, Vector s, Vector ds, int sx, int sy, int k0) {
	int k = get_global_id(0) + k0;
	int i, z, j;
	int sxy = sx * sy;
	KTensorRemap2D(k, z, i, sxy);
	REAL yi = s[k];
	REAL soma = 0.0;
	for (j = 0; j < sxy; ++j) {
		if (j == i) {
			soma += yi * (1 - yi) * ds[j + z * sxy];
//			printf("v(%d,%d,%d) =  %f, %f %f;\n", i+1, j+1, z+1, yi * (1 - yi),s[j + z * sxy],yi);
		} else {
			soma += -yi * s[j + z * sxy] * ds[j + z * sxy];
//			printf("v(%d,%d,%d) =  %f, %f %f;\n", i+1, j+1, z+1, yi * -s[j + z * sxy],s[j + z * sxy],yi);
		}
	}
	da[k] = soma;
}
/**
 * @goal Calcular os gradientes de entrada
 * @iteration dimensão da entrada a(x,y,z)
 * @param da Tensor de gradientes de entrada (escrita)
 * @param ds Tensor gradiente da saída (leitura)
 * @param sx dimensão x da saída
 * @param sy dimensão y da saída
 * @param k0 usado internamente no kernel
 */
kV softMaxcalcgradWhenNorm(Vector da, Vector ds, __global int *i_max, int sx, int sy, int k0) {
	int k = get_global_id(0) + k0;
	int i, z, j;
	int sxy = sx * sy;
	KTensorRemap2D(k, z, i, sxy);
	REAL soma = 0.0;
	for (j = 0; j < sxy; ++j) {
		soma += ((i == j) - (j == i_max[z])) * ds[j + z * sxy];
	}
	da[k] = soma;
}

/**
 * @goal Encontrar o maximo e o indice de cada dimensão z
 * @iteration dimensão z da entrada a(:,:,z)
 * @param a entrada
 * @param mx tensor maximos
 * @param i_max tensor indice de maximos
 * @param ax entrada x
 * @param ay entrada y
 * @param k0 uso interno no kernel
 */
kV softmaxFindMax(Vector a, Vector mx, __global int *i_max, int ax, int ay, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL maximo = a[KTensorMap(0, 0, z, ax, ay)];
	REAL adata;
	int imax = 0;
	for (x = 0; x < ax; x++) {
		for (y = 0; y < ay; y++) {
			adata = a[KTensorMap(x, y, z, ax, ay)];
			if (maximo < adata) {
				maximo = adata;
				imax = x * ay + y;
			}
		}
	}
	i_max[z] = imax;
	mx[z] = maximo;
}

/**
 * @goal calcular e^(a(x,y,z) - max(a))
 * @iteration dimensão de a (x,y,z)
 * @param entrada Tensor de entrada (leitura)
 * @param exponent Tensor e^entrada (escrita)
 * @param mx tensor maximos
 * @param ax entrada x
 * @param ay entrada y
 * @param k0 usado internamente no kernel
 */
kV softmaxExpNorm(Vector entrada, Vector exponent, Vector mx, int ax, int ay, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	KTensorRemap(k, x, y, z, ax, ay);
	exponent[k] = EXP(entrada[k] - mx[z]);
}
