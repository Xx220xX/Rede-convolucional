//
// Created by Xx220xX on 10/05/2020.
//a
#ifndef CL_KERNEL_SRC_H
#define CL_KERNEL_SRC_H
//#define __kernel
//#define __global
//#define get_global_id(x)x

typedef double (*dfd)(double);

static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }

static double difsigmoid(double x) {
    double tmp = sigmoid(x);
    return tmp * (1 - tmp);
}

static double tanghG(double x) { return tanh(x); }

static double diftanhG(double x) {
    double tmp = tanh(x);
    return (1 - tmp * tmp);
}

static double relu(double x) { return x > 0 ? x : 0.0; }

static double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }

static dfd func[6] = {sigmoid, difsigmoid, tanghG, diftanhG, relu, difrelu};

void initFucntions() {
    func[0] = (void *) sigmoid;
    func[1] = (void *) difsigmoid;
    func[2] = (void *) tanghG;
    func[3] = (void *) diftanhG;
    func[4] = (void *) relu;
    func[5] = (void *) difrelu;
}

#define TensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(y)*tx+(x))
typedef struct {
    int x, y, z;
} Ponto3d;
typedef struct {
    Ponto3d min, max;
} Range;

__kernel void convSum(__global double *filtro, __global double *entrada, __global double *saida,
                      int filtrok, int passo, int saidatx, int saidaty, int entradatx, int entradaty,
                      int lenFilter, int filtroz, int k0) {
    int k = get_global_id(0) + k0;
    int y = k % saidaty;
    int x = k / saidaty;
    Ponto3d mapeado = {x * passo, y * passo, 0};
    double sum = 0, f, v;
    for (int i = 0; i < lenFilter; i++)
        for (int j = 0; j < lenFilter; j++)
            for (int z = 0; z < filtroz; z++) {
                f = filtro[TensorMap(i, j, z, lenFilter, lenFilter)];
                v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];
                sum += f * v;
            }
    saida[TensorMap(x, y, filtrok, saidatx, saidaty)] = sum;
}

__kernel void convFixWeight(__global double *filtro, __global double *grad, __global double *gradOld, double hitlearn,
                            double momento, double multp, double weightDecay, int k0) {
    int k = get_global_id(0) + k0;
    double m = grad[k] + gradOld[k] * momento;
    double w = filtro[k];
    filtro[k] = w - hitlearn * (m * multp + w * weightDecay);
    gradOld[k] = m;

}

int normaliza_range(double f, int max, int lim_min) {
    if (f < 0)return 0;
    if (f >= max - 1)return max - 1;
    if (lim_min) return ceil(f);
    else return floor(f);
}

Range mapeia_entrada_saida(int x, int y, int passo, int tamanhoFiltro, int saidatx, int saidaty, int numeroFiltros) {
    double a = x, b = y;
    Range r = {0};
    r.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saidatx, 1);
    r.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saidaty, 1);
    r.max.x = normaliza_range(a / passo, saidatx, 0);
    r.max.y = normaliza_range(b / passo, saidaty, 0);
    r.max.z = numeroFiltros - 1;
    return r;
}

__kernel void convCalcGrads(__global double **filtros, __global double **gradFiltros, __global double *entrada, __global double *gradEntrada,
                            __global double *gradNext, int lenFilter, int passo, int entradatx, int entradaty, int saidatx, int saidaty,
                            int numFilters, int k0) {
    int k = get_global_id(0) + k0;
    int x = k % entradatx;
    int y = (k - x) % (entradatx * entradaty);
    int z = (k - y * entradatx - x) / (entradatx * entradaty);
    Range range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, numFilters);
    int minX, minY;
    double somaErro = 0, pesoAplicado = 0;
    for (int i = range.min.x; i <= range.max.x; i++) {
        minX = i * passo;
        for (int j = range.min.y; j <= range.max.y; j++) {
            minY = j * passo;
            for (int l = range.min.z; l <= range.max.z; l++) {
                pesoAplicado = filtros[l][TensorMap(x - minX, y - minY, z, lenFilter, lenFilter)];
                somaErro += pesoAplicado * gradNext[TensorMap(i, j, l, saidatx, saidaty)];
                gradFiltros[l][TensorMap(x - minX, y - minY, z, lenFilter, lenFilter)] += entrada[k] * gradNext[TensorMap(i, j, l, saidatx, saidaty)];
            }
        }
    }
    gradEntrada[k] = somaErro;
}


__kernel void fullfeed(__global double *entrada, __global double *pesos, __global double *input, __global double *saida, int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
    int n = get_global_id(0) + k0;
    double valorEntrada = 0;
    int m;
    for (int x = 0; x < inx; x++)
        for (int y = 0; y < iny; y++)
            for (int z = 0; z < inz; z++) {
                m = TensorMap(x, y, z, inx, iny);//z * (inx *iny) + y * inx + x;
                valorEntrada += entrada[m] * pesos[TensorMap(m, n, 0, pesosx, pesosy);
            }
    input[n] = valorEntrada;
    saida[n] = func[funcaoativacao](valorEntrada);
}

__kernel void fullfixweight(__global double *entrada, __global double *pesos, __global double *grad, __global double *oldgrad,
                            double hitlearn, double decaimentoDePeso,
                            int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
    int n = get_global_id(0) + k0;

    double tmp = grad[n] + oldgrad[n] * momento;

    for (int i = 0; i < inx; ++i) {
        for (int j = 0; j < iny; ++j) {
            for (int z = 0; z < inz; ++z) {
                m = TensorMap(i, j, z, inx, iny);
                w = pesos[TensorMap(m, n, 0, pesosx, pesosy);
                w -= hitLearn * (tmp * entrada[TensorAT(i, j, z, inx, iny)] + w * decaimentoDePeso);
                pesos[TensorMap(m, n, 0, pesosx, pesosy)] = w;
            }
        }
    }
    oldgrad[n] = tmp;
}

__kernel void fullcalcgrads1(__global double *grad, __global double *gradNext, __global double *input, int dfa,int k0) {
    int n = get_global_id(0) + k0;
    grad[n] = gradNext[n] * func[dfa](input[n]);
}

__kernel void fullcalcgrads2(__global double *grad,__global double * gradsEntrada,__global double * pesos,int pesosx, int pesosy,int k0) {
    int m = get_global_id(0) + k0;
    gradsEntrada[m] = 0;
    for (int n = 0; n < pesosy; ++n) {
        gradsEntrada[m] += grad[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];
    }
}

#endif //CL_TESTE_KERNEL_SRC_H
