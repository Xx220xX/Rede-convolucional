kV BatchNormMedia(Vr a, Vw u, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	int x, y;
	REAL m = 0;
	for (x = 0; x < ax; x++) {
		for (y = 0; y < ay; y++) {
			m += a[kMap(x, y, z, ax, ay)];
		}
	}
	u[z] = m / (REAL) (ax * ay);
}

kV BatchNormInvDesv(Vr a, Vr u, Vr o, REAL episolon, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	REAL sum = 0;
	REAL tmp;
	for (int x = 0; x < ax; x++) {
		for (int y = 0; y < ay; y++) {
			tmp = (a[kMap(x, y, z, ax, ay)] - u[z]);
			sum += tmp * tmp;
		}
	}
	sum = sum / (ax * ay);
	o[z] = 1.0 / sqrt(sum + episolon);
}

kV BatchNormNormaliza(Vw s, Vrw v, Vr a, Vr u, Vr o, Vr Y, Vr B, int ax, int ay, int id_0) {
	int x, y, z;
	int k = get_global_id(0) + id_0;
	kRap(k, x, y, z, ax, ay)
	v[k] = (a[k] - u[z]) * o[z];
	s[k] = v[k] * Y[z] + B[z];
}

kV BatchNormaCalcDnorm(Vw dv, Vr ds,Vr Y, int ax, int ay, int id_0) {
	int x, y, z;
	int k = get_global_id(0) + id_0;
	kRap(k, x, y, z, ax, ay)
	dv[k] = ds[k] * Y[z];
}

kV BatchNormMediadnorm_norma(Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	int x, y;
	REAL md = 0;
	REAL m = 0;
	for (x = 0; x < ax; x++) {
		for (y = 0; y < ay; y++) {
			m += dv[kMap(x, y, z, ax, ay)];
			md += (dv[kMap(x, y, z, ax, ay)] * v[kMap(x, y, z, ax, ay)]);
		}
	}
	mdnorm[z] = m / (REAL) (ax * ay);
	mdnormnorm[z] = md / (REAL) (ax * ay);
}

kV BatchNormaCalcDa(Vr da, Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, Vr o, int ax, int ay, int id_0) {
	int x, y, z;
	int k = get_global_id(0) + id_0;
	kRap(k, x, y, z, ax, ay)
	da[k] = o[z] * (dv[k] - mdnorm[z] - v[k] * mdnormnorm[z]);
}

kV BatchNormaCalcdYdB(Vr ds, Vr v, Vw dY, Vw dB, long batchSize, int ax, int ay, int id_0) {
	int z = get_global_id(0) + id_0;
	REAL sumY = 0;
	REAL sumB = 0;
	int k;
	for (int x = 0; x < ax; ++x) {
		for (int y = 0; y < ay; ++y) {
			k = kMap(x, y, z, ax, ay);
			sumB += ds[k];
			sumY += ds[k] * v[k];
		}
	}
	dB[z] = dB[z] + sumB / (REAL) batchSize;
	dY[z] = dY[z] + sumY / (REAL) batchSize;
}

kV BatchNormaLearn(Vrw Y, Vrw B, Vrw dY, Vrw dB, REAL hit, REAL momento, REAL decaimento, int id_0) {
	int z = get_global_id(0) + id_0;
	Y[z] = Y[z] - hit * (dY[z] + Y[z] * decaimento);
	B[z] = B[z] - hit * (dB[z] + B[z] * decaimento);
	dY[z] = dY[z] * momento;
	dB[z] = dB[z] * momento;
}