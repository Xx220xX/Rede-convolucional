
/// achar a media
/// ativa 1
kV BatchNormMedia(Vector entrada, Vector media,
				  int entradatx, int entradaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL m = 0;
	for (x = 0; x < entradatx; x++) {
		for (y = 0; y < entradaty; y++) {
			m += entrada[KTensorMap(x, y, z, entradatx, entradaty)];
		}
	}
	media[z] = m / (REAL)(entradatx * entradaty);
}

/// achar a diferenca
/// ativa 2
kV BatchNormDiferenca(Vector entrada, Vector media,
					  Vector diferenca,
					  Vector diferencaquad,
					  int entradatx, int entradaty, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, entradatx, entradaty)
	diferenca[k] = entrada[k] - media[z];
	diferencaquad[k] = diferenca[k] * diferenca[k];
}
/// ativa 3

kV BatchNormVariance(Vector dif, Vector difQuad,
					 Vector sumdiferenca, Vector variancia,
					 REAL episolon, int diftx, int difty,
					 int k0) {
	int z = get_global_id(0) + k0;
	REAL sum = 0;
	REAL sumdif = 0;
	for (int x = 0; x < diftx; x++) {
		for (int y = 0; y < difty; y++) {
			sum += difQuad[KTensorMap(x, y, z, diftx, difty)];
			sumdif += dif[KTensorMap(x, y, z, diftx, difty)];
		}
	}
	sumdiferenca[z] = sumdif;
	variancia[z] = sqrt(sum / (difty * diftx) + episolon);
}

/// normaliza
/// ativa 4

kV BatchNormNormaliza(Vector saida,
					  Vector norma,
					  Vector diferenca,
					  Vector variancia,
					  Vector Y,
					  Vector B,
					  int diferencatx, int diferencaty, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, diferencatx, diferencaty)
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
	KTensorRemap(k, x, y, z, entradatx, entradaty)
	REAL M = entradatx * entradaty;
	REAL dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);
	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);

	REAL didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;
	didx = didx / (variancia[z] * variancia[z]);
	didx = didx * gradNext[k];
	gradIn[k] = didx * Y[z];
}
kV BatchNormaCalcGrad2(Vector gradNext,
					   Vector norma,
					   Vector Y,
					   Vector B,
					   Vector gradY,
					   Vector gradB,
					   REAL hitlearn,
					   REAL momento,
					   REAL weightDecay,
					   int entradatx,
					   int entradaty,
					   int k0) {
	int z = get_global_id(0) + k0;
	REAL sumY = 0;
	REAL sumB = 0;
	int k;
	for (int x = 0; x < entradatx; ++x) {
		for (int y = 0; y < entradaty; ++y) {
			k = KTensorMap(x, y, z, entradatx, entradaty);
			sumY += gradNext[k];
			sumB += gradNext[k] * norma[k];
		}
	}
	gradB[z] = sumB + gradB[z] * momento;
	gradY[z] = sumY + gradY[z] * momento;

	B[z] = B[z] - (gradB[z] + weightDecay * B[z]) * hitlearn;
	Y[z] = Y[z] - (gradY[z] + weightDecay * Y[z]) * hitlearn;
}


