
// achar a media
kV BatchNormMedia(Vector entrada, Vector media,
                  int entradatx, int entradaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	double m = 0;
	for (x = 0; x < entradatx; x++) {
		for (y = 0; y < entradaty; y++) {
			m += entrada[TensorMap(x, y, z, entradatx, entradaty)];
		}
	}
	media[z] = m / (double) (entradatx * entradaty);
}

// achar a diferenca
kV BatchNormDiferenca(Vector entrada, Vector media,
                      Vector diferenca,
                      Vector diferencaquad,
                      int entradatx, int entradaty, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	diferenca[k] = entrada[k] - media[z];
	diferencaquad[k] = diferenca[k] * diferenca[k];
}

kV BatchNormVariance(Vector dif, Vector difQuad,
                     Vector sumdiferenca, Vector variancia,
                     double episolon, int diftx, int difty,
                     int k0) {
	int z = get_global_id(0) + k0;
	double sum = 0;
	double sumdif = 0;
	for (int x = 0; x < diftx; x++) {
		for (int y = 0; y < difty; y++) {
			sum += difQuad[TensorMap(x, y, z, diftx, difty)];
			sumdif += dif[TensorMap(x, y, z, diftx, difty)];
		}
	}
	sumdiferenca[z] = sumdif;
	variancia[z] = sqrt(sum / (difty * diftx) + episolon);
}

// normaliza
kV BatchNormNormaliza(Vector saida,
                      Vector norma,
                      Vector diferenca,
                      Vector variancia,
                      Vector Y,
                      Vector B,
                      int diferencatx, int diferencaty, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	TensorRemap(k, x, y, z, diferencatx, diferencaty)
	norma[k] = diferenca[k] / variancia[z];
	saida[k] = norma[k] * Y[z] + B[z];
}


kV BatchNormaCalcGrad1(Vector gradIn,
                       Vector gradNext,
                       Vector variancia,
                       Vector media,
                       Vector Y,

                       Vector somaDif,
                       Vector entrada,
                       int entradatx,
                       int entradaty,
                       int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	TensorRemap(k, x, y, z, entradatx, entradaty)
	double M = entradatx * entradaty;
	double dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);
	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);

	double didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;
	didx = didx / (variancia[z] * variancia[z]);
	didx = didx * gradNext[k];
	gradIn[k] = didx * Y[z];
}

kV BatchNormaCalcGrad2(Vector gradNext,
                       Vector norma,
                       Vector gradY,
                       Vector gradB,
                       int entradatx,
                       int entradaty,
                       int k0) {
	int z = get_global_id(0) + k0;
	double sumY = 0;
	double sumB = 0;
	int k;
	for (int x = 0; x < entradatx; ++x) {
		for (int y = 0; y < entradaty; ++y) {
			k = TensorMap(x, y, z, entradatx, entradaty);
			sumY += gradNext[k];
			sumB += gradNext[k] * norma[k];
		}
	}
	gradB[z] = sumB;
	gradY[z] = sumY;
}


kV batchNormCorrigePeso(Vector gradY,
                        Vector gradB,
                        Vector Y,
                        Vector B,
                        double hitlearn,
                        int k0) {
	int z = get_global_id(0) + k0;
	B[z] = B[z] - gradB[z] * hitlearn;
	Y[z] = Y[z] - gradY[z] * hitlearn;
}