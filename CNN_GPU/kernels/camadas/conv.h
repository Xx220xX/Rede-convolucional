//#include"utils.h"
kV convSum(Vector filtro, Vector entrada, Vector saida,
           int passo, int saidatx, int saidaty, int entradatx, int entradaty,
           int lenFilter, int entradatz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	TensorRemap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d mapeado = {x * passo, y * passo, 0};
	double sum = 0, f, v;
	for (int m = 0; m < lenFilter; m++)
		for (int n = 0; n < lenFilter; n++)
			for (int z = 0; z < entradatz; z++) {
				f = filtro[TensorMap4D(m, n, z, filtrok, lenFilter, lenFilter, entradatz)];
				v = entrada[TensorMap(mapeado.x + m, mapeado.y + n, z, entradatx, entradaty)];
				sum += f * v;
			}
	saida[k] = sum;

}

kV convFixWeight(Vector filtro, Vector grad, Vector gradOld, double hitlearn,
                 double momento, double weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	double m = grad[k] + gradOld[k] * momento;
	double w = filtro[k];
	filtro[k] = w - hitlearn * (m + w * weightDecay);
	gradOld[k] = m;
}

kV convCalcFiltro(     Vector ds,
					   Vector entrada,
					   Vector gradFiltro,
                       int gradFiltro_tx,
                       int gradFiltro_ty,
                       int gradFiltro_tz,
                       int entrada_tx,
                       int entrada_ty,
                       int saida_tx,
                       int saida_ty,
                       int passo,
                       int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
//	printf("kernel %d\n",k);
	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)
	double soma = 0,aux;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			aux = entrada[TensorMap(i*passo+m, j*passo+n,z,entrada_tx,entrada_ty)]
				   *ds[TensorMap(i,j,l,saida_tx,saida_ty)];
			//aux = (!(isnan(aux) || isinf(aux)))*aux;
			soma +=aux;
		}
	}
	gradFiltro[k] = soma;
}

kV convCalcGrads(Vector filtro,
				 Vector entrada,
                 Vector gradEntrada,
                 Vector gradNext,
                 int lenFilter,
                 int filtroz,
                 int passo,
                 int entradatx,
                 int entradaty,
                 int saidatx,
                 int saidaty,
                 int numFilters,
                 int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	Range range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, numFilters);
	int minX, minY;
	double somaErro = 0, pesoAplicado = 0;
	double aux;
	for (int i = range.min.x; i <= range.max.x; i++) {
		minX = i * passo;
		for (int j = range.min.y; j <= range.max.y; j++) {
			minY = j * passo;
			for (int l = range.min.z; l <= range.max.z; l++) {
				pesoAplicado = filtro[TensorMap4D(x - minX, y - minY, z, l, lenFilter, lenFilter, filtroz)];
				aux = pesoAplicado * gradNext[TensorMap(i, j, l, saidatx, saidaty)];
				//aux = (!(isnan(aux) || isinf(aux)))*aux;
				somaErro +=aux;
			}
		}
	}
	gradEntrada[k] = somaErro;
}

