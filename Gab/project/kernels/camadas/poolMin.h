kV poolativaMin(Vr A, Vr S, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, saidatx, saidaty)

	Ponto3d mapeado = {x * passox, y * passoy, 0};
	REAL mval, v;
	mval = DBL_MAX;
	for (int i = 0; i < filtrox; ++i) {
		for (int j = 0; j < filtroy; ++j) {
			v = A[kMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
			if (v < mval) {
				mval = v;
			}
		}
	}
	S[k] = mval;
}

