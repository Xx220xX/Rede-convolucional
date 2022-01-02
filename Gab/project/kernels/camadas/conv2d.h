kV conv2dSum(Vr W, Vr a, Vw Z, Vw s, int px, int py, int sx, int sy, int ax, int ay, int az, int fx, int fy, int fz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, zl;
	int z, l;
	kRap(k, x, y, zl, sx, sy)
	l = zl / az;
	z = zl % az;
	REAL sum = 0, f, v;
	int lf, le;
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			lf = kMap(m, n, l, fx, fy);
			le = kMap(x * px + m, y * py + n, z, ax, ay);
			f = W[lf];
			v = a[le];
			sum += f * v;
		}
	}
	Z[k] = sum;
	s[k] = func(fid, sum);
}

kV conv2dCalcGradZ(Vr ds, Vr z, Vw dz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	dz[k] = ds[k] * func(fid, z[k]);
}

kV conv2dCalcGradIn(Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int az, int sx, int sy,  int k0) {
	int k = get_global_id(0) + k0;
	int x, y, z;
	kRap(k, x, y, z, atx, aty)
	Range range_W;
	range_W.min.x = 0;
	if (x + fx > atx) {
		range_W.min.x = x + fx - atx;
	}
	range_W.max.x = fx - 1;
	if (x - fx + 1 < 0) {
		range_W.max.x = x;
	}
	range_W.min.y = 0;
	if (y + fy > aty) {
		range_W.min.y = y + fy - aty;
	}
	range_W.max.y = fy - 1;
	if (y - fy + 1 < 0) {
		range_W.max.y = y;
	}
	REAL somaErro = 0, pesoAplicado = 0;
	int i, j;
	int lf, ls;
	for (int m = range_W.min.x; m <= range_W.max.x; m++) {
		i = (x - m) / px;
		if (i * px + m != x) {
			continue;
		}
		for (int n = range_W.min.y; n <= range_W.max.y; n++) {
			j = (y - n) / py;
			if (j * py + n != y) {
				continue;
			}
			for (int l = 0; l < fz; l++) {
				lf = kMap(m, n, l, fx, fy);
				ls = kMap(i, j, l * az + z, sx, sy);
				pesoAplicado = W[lf];
				somaErro += pesoAplicado * dz[ls];
			}
		}
	}
	da[k] = somaErro;
}

kV conv2dCalcGradAndFixWeight(Vrw W, Vr dz, Vr a, Vrw dW, int fx, int fy, int ax, int ay, int az, int sx, int sy, int px, int py, REAL hitLearn, REAL momento, REAL weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, l;
	kRap(k, m, n, l, fx, fy)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < sx; ++i) {
		for (int j = 0; j < sy; ++j) {
			for (int z = 0; z < az; ++z) {
				ls = kMap(i, j, l * az + z, sx, sy);
				le = kMap(i * px + m, j * py + n, z, ax, ay);
				soma += a[le] * dz[ls];
			}
		}
	}
	REAL dw = soma + dW[k] * momento;
	REAL w = W[k];
	W[k] = w - hitLearn * (dw + w * weightDecay);
	dW[k] = dw;
}

kV conv2dCalcGradBatch(Vr dz, Vr a, Vr dW, long batchSize, int fx, int fy, int fz, int ax, int ay, int az, int sx, int sy, int px, int py, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, l;
	kRap(k, m, n, l, fx, fy)
	REAL soma = 0;
	int l_a, l_dz;
	for (int i = 0; i < sx; ++i) {
		for (int j = 0; j < sy; ++j) {
			for (int z = 0; z < az; ++z) {
				l_dz = kMap(i, j, l * az + z, sx, sy);
				l_a = kMap(i * px + m, j * py + n, z, ax, ay);
				soma += a[l_a] * dz[l_dz];
			}
		}
	}
	soma = soma / batchSize + dW[k];
	dW[k] = soma;
}

