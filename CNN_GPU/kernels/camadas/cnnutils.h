//
// Created by Henrique on 22-Jul-21.
//


kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j, z;
	KTensorRemap(k, i, j, z, vx, vy)
	imi = imi + i;
	int imj = j + z * vy + z;
	out[imi * imy + imj] = ((int) v[k]) & 0xff;
}


kV normalizeVector(Vector input, Vector saida, REAL multiplicador, REAL somador, REAL subtrator,
				int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}


kV subKernel(Vector grad, Vector saida, Vector target, int k0) {
	int k = get_global_id(0) + k0;
	grad[k] = saida[k] - target[k];
}

kV divKernel(Vector v, REAL value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = v[k] / value;
}

kV divIntDo(__global unsigned char *src, Vector v, REAL value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = ((REAL) src[k]) / value;

}

kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {
	int w = get_global_id(0) + k0;
	int y = ints[w];
	v[KTensorMap4D(0,  y,0, w, 1, noptiobs, 1)] = 1.0;
}
