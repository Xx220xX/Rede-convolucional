//
// Created by Henrique on 22-Jul-21.
//


kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) {
	int k = get_global_id(0) + k0;
	int i, j, z;
	TensorRemap(k, i, j, z, vx, vy)
	imi = imi + i;
	int imj = j + z * vy + z;
	out[imi * imy + imj] = ((int) v[k]) & 0xff;
}


kV
normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,
				int k0) {
	int k = get_global_id(0) + k0;
	saida[k] = (input[k] + somador) * multiplicador - subtrator;
}


kV subKernel(Vector grad, Vector saida, Vector target, int k0) {
	int k = get_global_id(0) + k0;
	grad[k] = saida[k] - target[k];
}

kV divKernel(Vector v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = v[k] / value;
}

kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) {
	int k = get_global_id(0) + k0;
	v[k] = ((double) src[k]) / value;
}

kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {
//	int k = get_global_id(0) + k0;
//
//	int d;
//	for (int j = 0; j < noptiobs; j++) {
//		d = TensorMap4D(0, j, 0, k, 1, noptiobs, 1);
//		v[d] = (double) (j == ints[k]);
//	}
	int k = get_global_id(0) + k0;
	int x,y,z,w;
	TensorRemap4D(k,x,y,z,w,1,noptiobs,1);
	v[k] = (double) (y == ints[w]);
}



