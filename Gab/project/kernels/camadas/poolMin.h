kV poolativaMin(Vr A, Vr S, __global int * hmap, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = A[kMap(mapeado.x , mapeado.y , z, entradatx, entradaty)];
	int index,mn=0;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			index = kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty);
			v = A[index];
			if (v < mval) {
				mval = v;
				mn = index;
			}
		}
	}
	S[k] = mval;
	hmap[k] = mn;
}

