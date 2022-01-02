kV paddingfeed(Vr in, Vr out, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, txi, tyi)
	int s = kMap(x + t, y + l, z, txo, tyo);
	out[s] = in[k];
}

kV paddingBack(Vr gradNext, Vr gradin, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, txi, tyi)
	int s = kMap(x + t, y + l, z, txo, tyo);
	gradin[k] = gradNext[s];
}