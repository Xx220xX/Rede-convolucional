
__kernel void reluativa(__global double *entrada,__global double *saida, int k0){
     int k = get_global_id(0) + k0;
     double v = entrada[k];
     if (v < 0)
        v = 0;
     saida[k] = v;
}

__kernel void relucalcgrad(__global double *gradentrada,__global double *entrada, __global double *gradnext, int k0){
     int k = get_global_id(0) + k0;
     gradentrada[k] = entrada[k] ? (0) : (1*gradnext[k]);
}

__kernel void poolativa(__global double *entrada, __global double *saida, int lenFilter,
                      int passo, int saidatx, int saidaty, int saidatz, int k0) {
    int k = get_global_id(0) + k0;
    int y = k % saidaty;
    int x = k / saidaty;
    Ponto3d mapeado = {x * passo, y * passo, 0};
    double mval, v;
    mval = -DBL_MAX;
    for(int z = 0; z < saidatz; ++z){
        for (int i = 0; i < lenFilter; ++i){
            for (int j = 0; j < lenFilter; ++j){
                v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
                if (v > mval)
                    mval = v;
            }
        }
        saida[TensorMap(x, y, z, saidatx, saidaty)] = mval;
    }
}

Range mapeia_entrada_saidaPool(int x, int y, int tamanhoFiltro, int passo, int saidatx, int saidaty) {
    float a = x, b = y;
    Range g = {0};
    g.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saida->tx, 1);
    g.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saida->ty, 1);
    g.max.x = normaliza_range(a / passo, saidatx, 0);
    g.max.y = normaliza_range(b / passo, saidaty, 0);
    g.max.z = saida->tz - 1;
    return g;
}

__kernel void poolCalcGrads(__global double *entrada, __global double *gradEntrada, __global double *gradNext, __global double *saida,
                            int lenFilter, int passo, int entradatx, int entradaty, int saidatx, int saidaty, int k0) {
    int k = get_global_id(0) + k0;
    int x = k % entradatx;
    int y = (k - x) % (entradatx * entradaty);
    int z = (k - y * entradatx - x) / (entradatx * entradaty);
    Range range = mapeia_entrada_saidaPool(x, y, passo, lenFilter, saidatx, saidaty);
    int minX, minY;
    double somaErro = 0, testeMax;
    for (int i = range.min.x; i <= range.max.x; i++) {
        minX = i * passo;
        for (int j = range.min.y; j <= range.max.y; j++) {
            minY = j * passo;
            testeMax = entrada[k] == saida[TensorMap(i, j, z, saidatx, saidaty)];
            somaErro += testeMax * gradNext[TensorMap(i, j, z, saidatx, saidaty)];
        }
    }
    gradEntrada[k] = somaErro;
}

