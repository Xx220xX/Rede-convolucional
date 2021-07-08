//#include"utils.h"
kV convncSum(Vector filtro, Vector entrada, Vector saida,
             int passox, int passoy, int largx,
             int largy, int saidatx, int saidaty,
             int entradatx, int entradaty,int fx, int fy,
             int entradatz, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, filtrok;
	TensorRemap(k, x, y, filtrok, saidatx, saidaty)
	Ponto3d mapeado = {x * passox, y * passoy, 0};
	double sum = 0, f, v;
	for (int i = 0; i < fx; i++)
		for (int j = 0; j < fy; j++)
			for (int z = 0; z < entradatz; z++) {
				f = filtro[TensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];
				v = entrada[TensorMap(mapeado.x + i * largx, mapeado.y + j * largy, z, entradatx, entradaty)];

				sum += f * v;
			}
	saida[k] = sum;
}

kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,
				   double hitlearn,
                   double momento, double weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	double m = grad[k] + gradOld[k] * momento;
	double w = filtro[k];
	filtro[k] = w - hitlearn * (m + w * weightDecay);
	gradOld[k] = m;
}

kV convncCalcFiltro(Vector ds,
                    Vector entrada,
                    Vector gradFiltro,
                    int gradFiltro_tx,
                    int gradFiltro_ty,
                    int gradFiltro_tz,

                    int entrada_tx,
                    int entrada_ty,

                    int saida_tx,
                    int saida_ty,

                    int passox,
                    int passoy,

                    int largx,
                    int largy,
                    int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)
	double soma = 0,aux;
	for (int i = 0; i < saida_tx; ++i) {
		for (int j = 0; j < saida_ty; ++j) {
			aux = entrada[TensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]
			        * ds[TensorMap(i, j, l, saida_tx, saida_ty)];
			aux = (!(isnan(aux) || isinf(aux)))*aux;
			soma += aux;
		}
	}
	gradFiltro[k] = soma;
}

/**
 * equacao a ser implementada
 * x = s*p + m*w
 * onde:
 * 	x é da entrada 
 * 	s é da saida
 * 	m é do filtro
 * 	s = (x - m*w)/p
 */
kV convncCalcGrads(Vector filtro,
                   Vector entrada,
                   Vector gradEntrada,
                   Vector gradNext,

                   int passox,
                   int passoy,
                   int largx,
                   int largy,

                   int entradatx,
                   int entradaty,
                   int saidatx,
                   int saidaty,

                   int fx,
                   int fy,
                   int fz,
                   int numFilters,

                   int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	Range range_filtro ;
	range_filtro.min.x = 0;
	if ((entradatx - x - (fx - 1) * largx) < 0) {
		range_filtro.min.x = -entradatx + x + fx;
	}
	range_filtro.max.x = fx - 1;
	if (x - (fx - 1) * largx < 0) {
		range_filtro.max.x = x / largx;
	}
	range_filtro.min.y = 0;
	if ((entradaty - y - (fy - 1) * largy) < 0) {
		range_filtro.min.y = -entradaty + y + fy;
	}
	range_filtro.max.y = fy - 1;
	if (y - (fy - 1) * largy < 0) {
		range_filtro.max.y = y / largy;
	}
	int sx, sy;
	double somaErro = 0,aux, pesoAplicado = 0;
	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {
		sx = (x - m * largx) / passox;
		if (sx * passox + m * largx != x)continue;
		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {
			sy = (y - n * largy) / passox;
			if (sy * passoy + n * largy != y)continue;
			for (int l = 0; l < fz; l++) {
				pesoAplicado = filtro[TensorMap4D(m, n, z, l, fx, fy, fz)];
				aux = pesoAplicado * gradNext[TensorMap(sx, sy, l, saidatx, saidaty)];
				aux = (!(isnan(aux) || isinf(aux)))*aux;
				somaErro +=aux;
			}
		}
	}
	gradEntrada[k] = somaErro;
}

