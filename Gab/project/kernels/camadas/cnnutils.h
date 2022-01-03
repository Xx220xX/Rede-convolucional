//
// Created by Henrique on 22-Jul-21.
//


kV createImg(__global unsigned char *out, Vr v, int vx, int vy, int imi, int imy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j, z;
	kRap(k, i, j, z, vx, vy)
	imi = imi + i;
	int imj = j + z * vy + z;
	out[imi * imy + imj] = ((int) v[k]) & 0xff;
}

kV putIMG(__global unsigned char *imagem_saida, Vr v, int z, REAL px, REAL py, int imy, int width, int i0, int j0, int vx, int vy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j;
	KRap2D(k, i, j, imy)
	int x = i * px, y = j * py;
	imagem_saida[(i + i0) * width + j + j0] = ((int) v[kMap(x, y, z, vx, vy)]) & 0xff;
}


kV normalizeVector(Vr input, Vr saida, REAL multiplicador, REAL somador, REAL subtrator, int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}


kV kernel_sub(Vr ds, Vr s, Vr t, int k0) {
	int k = get_global_id(0) + k0;
	ds[k] = s[k] - t[k];
}

kV kernel_normalizechar2real(Vr dst, __global unsigned char *src, REAL a, REAL b, int k0) {
	int k = get_global_id(0) + k0;
//	printf("update\n");
	dst[k] = ((REAL) src[k] - b) / a;
}

kV kernel_getVetorClassFromChar(Vr dst, __global unsigned char *ints, unsigned int noptiobs, int k0) {
	int w = get_global_id(0) + k0;
	int y = ints[w];
	dst[kMap4D(0, y, 0, w, 1, noptiobs, 1)] = 1.0;
}
kV kernel_fixW(Vr w, Vr dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0) {
	int k = get_global_id(0) + k0;
	w[k] = w[k] - hitlearn * dw[k] -  hitlearn * w[k] * decaimentoDePeso ;
	dw[k] = dw[k] * momento;
}