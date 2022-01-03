kV convFSum(Vr W, Vr B, Vr A, Vw Z, Vw S, int px, int py, int sx, int sy, int atx, int aty, int fx, int fy, int fz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	int x, y, w;

	kRap(k, x, y, w, sx, sy)
	REAL sum, f, v;
	int lf, le;
	sum = B[w];
	for (int m = 0; m < fx; m++) {
		for (int n = 0; n < fy; n++) {
			for (int z = 0; z < fz; z++) {
				lf = kMap4D(m, n, z, w, fx, fy, fz);
				le = kMap(x * px + m, y * py + n, z, atx, aty);
				f = W[lf];
				v = A[le];
				sum += f * v;
			}
		}
	}
	Z[k] = sum;
	S[k] = func(fid, sum);
}

kV convFCalcGradZ(Vr ds, Vr z, Vw dz, int fid, int k0) {
	int k = get_global_id(0) + k0;
	dz[k] = ds[k] * func(fid, z[k]);
}

kV convFCalcGradBAndFix(Vrw B, Vrw dB, Vr dZ, int dzx, int dzy, REAL hitLearn, REAL momento, REAL weightDecay, int k0) {
	int w = get_global_id(0) + k0;
	REAL sum = 0;
	for (int x = 0; x < dzx; ++x) {
		for (int y = 0; y < dzy; ++y) {
			sum += dZ[kMap(x, y, w, dzx, dzy)];
		}
	}
	dB[w] = sum + dB[w] * momento;
	B[w] = B[w] - hitLearn * (dB[w] + B[w] * weightDecay);
}

kV convFCalcGradBBatch(Vrw dB, Vr dZ, int dzx, int dzy, long batchSize, int k0) {
	int w = get_global_id(0) + k0;
	REAL sum = 0;
	for (int x = 0; x < dzx; ++x) {
		for (int y = 0; y < dzy; ++y) {
			sum += dZ[kMap(x, y, w, dzx, dzy)];
		}
	}
	sum = sum / batchSize + dB[w];
	dB[w] = sum;

}


kV convFCalcGradIn(Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int sx, int sy, int sz, int k0) {
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
			for (int w = 0; w < sz; w++) {
				lf = kMap4D(m, n, z, w, fx, fy, fz);
				ls = kMap(i, j, w, sx, sy);
				pesoAplicado = W[lf];
				somaErro += pesoAplicado * dz[ls];
			}
		}
	}
	da[k] = somaErro;
}

kV convFCalcGradAndFixWeight(Vr W, Vr dz, Vr a, Vr dW, int fx, int fy, int fz, int a_tx, int a_ty, int s_tx, int s_ty, int px, int py, REAL hitLearn, REAL momento, REAL weightDecay, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < s_tx; ++i) {
		for (int j = 0; j < s_ty; ++j) {
			le = kMap(i * px + m, j * py + n, z, a_tx, a_ty);
			ls = kMap(i, j, l, s_tx, s_ty);
			soma += a[le] * dz[ls];
		}
	}
	REAL dw = soma + dW[k] * momento;
	REAL w = W[k];
	W[k] = w - hitLearn * (dw + w * weightDecay);
	dW[k] = dw;
}

kV convFCalcGradBatch(Vr dz, Vr a, Vr dW, long batchSize, int fx, int fy, int fz, int a_tx, int a_ty, int s_tx, int s_ty, int px, int py, int k0) {
	int k = get_global_id(0) + k0;
	int m, n, z, l;
	kRep4D(k, m, n, z, l, fx, fy, fz)
	REAL soma = 0;
	int le, ls;
	for (int i = 0; i < s_tx; ++i) {
		for (int j = 0; j < s_ty; ++j) {
			le = kMap(i * px + m, j * py + n, z, a_tx, a_ty);
			ls = kMap(i, j, l, s_tx, s_ty);
			soma += a[le] * dz[ls];
		}
	}
	soma = soma / batchSize + dW[k];
	dW[k] = soma;
}

