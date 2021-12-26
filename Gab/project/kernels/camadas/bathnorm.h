/// achar a media
/// ativa 1
kV BatchNormMedia(Vector entrada, Vector media, int entradatx, int entradaty, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL m = 0;
	for (x = 0; x < entradatx; x++) {
		for (y = 0; y < entradaty; y++) {
			m += entrada[KTensorMap(x, y, z, entradatx, entradaty)];
		}
	}
	media[z] = m / (REAL) (entradatx * entradaty);
}

kV BatchNormInvDesv(Vector a, Vector media, Vector inv_desv, REAL episolon, int ax, int ay, int k0) {
	int z = get_global_id(0) + k0;
	REAL sum = 0;
	REAL tmp;
	for (int x = 0; x < ax; x++) {
		for (int y = 0; y < ay; y++) {
			tmp = (a[KTensorMap(x, y, z, ax, ay)] - media[z]);
			sum += tmp * tmp;
		}
	}
	sum = sum / (ax * ay);
	inv_desv[z] = 1.0 / sqrt(sum + episolon);
}

kV BatchNormNormaliza(Vector saida, Vector norma, Vector a, Vector media, Vector inv_std, Vector Y, Vector B, int ax, int ay, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, ax, ay)
	norma[k] = (a[k] - media[z]) * inv_std[z];
	saida[k] = norma[k] * Y[z] + B[z];
}

kV BatchNormaCalcDnorm(Vector dnorm, Vector ds, Vector Y, int ax, int ay, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, ax, ay)
	dnorm[k] = ds[k] * Y[z];
}

kV BatchNormMediadnorm_norma(Vector norm, Vector dnorm, Vector mdnorm, Vector mdnormnorm, int ax, int ay, int k0) {
	int z = get_global_id(0) + k0;
	int x, y;
	REAL md = 0;
	REAL m = 0;
	REAL tmp;
	for (x = 0; x < ax; x++) {
		for (y = 0; y < ay; y++) {
			m += dnorm[KTensorMap(x, y, z, ax, ay)];
			md += (dnorm[KTensorMap(x, y, z, ax, ay)] * norm[KTensorMap(x, y, z, ax, ay)]);
		}
	}
	mdnorm[z] = m / (REAL) (ax * ay);
	mdnormnorm[z] = md / (REAL) (ax * ay);
}

kV BatchNormaCalcDa(Vector da, Vector norm, Vector dnorm, Vector mdnorm, Vector mdnormnorm, Vector inv_std, int ax, int ay, int k0) {
	int x, y, z;
	int k = get_global_id(0) + k0;
	KTensorRemap(k, x, y, z, ax, ay)
	da[k] = inv_std[z] * (dnorm[k] - mdnorm[z] - norm[k] * mdnormnorm[z]);
}


kV BatchNormaCalcdYdB(Vector ds, Vector norma, Vector gradY, Vector gradB, long batchSize, int ax, int ay, int k0) {
	int z = get_global_id(0) + k0;
	REAL sumY = 0;
	REAL sumB = 0;
	int k;
	for (int x = 0; x < ax; ++x) {
		for (int y = 0; y < ay; ++y) {
			k = KTensorMap(x, y, z, ax, ay);
			sumB += ds[k];
			sumY += ds[k] * norma[k];
		}
	}
//	printf("%d %f %f\n",z,gradB[z],gradY[z]);
	gradB[z] = gradB[z] + sumB / (REAL) batchSize;
	gradY[z] = gradY[z] + sumY / (REAL) batchSize;
}

kV BatchNormaLearn(Vector Y, Vector B, Vector gradY, Vector gradB, REAL hit, REAL momento, REAL decaimento, int k0) {
	int z = get_global_id(0) + k0;
	Y[z] = Y[z] - hit * (gradY[z] + Y[z] * decaimento);
	B[z] = B[z] - hit * (gradB[z] + B[z] * decaimento);

	gradY[z] = gradY[z] * momento;
	gradB[z] = gradB[z] * momento;
}


