//
// Created by Xx220xX on 10/05/2020.
//
#ifndef CL_KERNEL_SRC_H
#define CL_KERNEL_SRC_H

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
double func(int id,double  x){
    switch (id) {
        case 0:return sigmoid(x);
        case 1:return difsigmoid(x);
        case 2:return tanghG(x);
        case 3:return diftanhG(x);
        case 4:return relu(x);
        case 5:return difrelu(x);
        default: return 0;
    }
}

#define TensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(y)*tx+(x))
#define TensorMap4D(x, y, z,l, tx, ty,tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(y)*tx+(x))
typedef struct {
    int x, y, z;
} Ponto3d;
typedef struct {
    Ponto3d min, max;
} Range;


__kernel void printTensor(__global double *t,int mx,int my,int mz,int ofset){
    for (int z = 0; z < mz; z++) {
        printf("[Dim%d]\n", z);
        for (int x = 0; x < mx; x++) {
            for (int y = 0; y < my; y++) {

                printf("%.2lf \t",  t[TensorMap(x,y,z,mx,my)+ofset]);
            }
            printf("\n");
        }
    }
}
__kernel void convSum(__global double *filtro, __global double *entrada, __global double *saida,
                      int passo, int saidatx, int saidaty, int entradatx, int entradaty,
                      int lenFilter, int entradatz, int k0) {
    int k = get_global_id(0) + k0;
    int x = k% saidatx;
    int y = ((k  - x) % (saidaty*saidatx))/saidatx;
    int filtrok =  (k-y*saidatx-x)/(saidatx*saidaty);

    Ponto3d mapeado = {x * passo, y * passo, 0};
    double sum = 0, f, v;
    for (int i = 0; i < lenFilter; i++)
        for (int j = 0; j < lenFilter; j++)
            for (int z = 0; z < entradatz; z++) {
                f = filtro[TensorMap4D(i,j,z,filtrok,lenFilter,lenFilter,entradatz)];
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
    Range r ;
    r.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saidatx, 1);
    r.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saidaty, 1);
    r.min.z = 0;

    r.max.x = normaliza_range(a / passo, saidatx, 0);
    r.max.y = normaliza_range(b / passo, saidaty, 0);
    r.max.z = numeroFiltros - 1;
    return r;
}

__kernel void convCalcGrads(__global double *filtro, __global double *gradFiltro, __global double *entrada, __global double *gradEntrada,
                            __global double *gradNext, int lenFilter,int filtroz, int passo, int entradatx, int entradaty, int saidatx, int saidaty,
                            int numFilters, int k0) {
    int k = get_global_id(0) + k0;
    int x = k % entradatx;
    int y = ((k - x) % (entradatx * entradaty))/entradatx;
    int z = (k - y * entradatx - x) / (entradatx * entradaty);
    Range range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, numFilters);
    int minX, minY;
    double somaErro = 0, pesoAplicado = 0;
    for (int i = range.min.x; i <= range.max.x; i++) {
        minX = i * passo;
        for (int j = range.min.y; j <= range.max.y; j++) {
            minY = j * passo;
            for (int l = range.min.z; l <= range.max.z; l++) {
                pesoAplicado = filtro[TensorMap4D(x - minX, y - minY, z,l, lenFilter, lenFilter,filtroz)];
                somaErro += pesoAplicado * gradNext[TensorMap(i, j, l, saidatx, saidaty)];
                gradFiltro[TensorMap4D(x - minX, y - minY, z,l, lenFilter, lenFilter,filtroz)] += entrada[k] * gradNext[TensorMap(i, j, l, saidatx, saidaty)];
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
                valorEntrada += entrada[m] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];
            }
    input[n] = valorEntrada;
    saida[n] = func(funcaoativacao,valorEntrada);
}

__kernel void fullfixweight(__global double *entrada, __global double *pesos, __global double *grad, __global double *oldgrad,
                            double hitlearn, double decaimentoDePeso,double momento,
                            int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
    int n = get_global_id(0) + k0;
    int m;
    double w;
    double tmp = grad[n] + oldgrad[n] * momento;

    for (int i = 0; i < inx; ++i) {
        for (int j = 0; j < iny; ++j) {
            for (int z = 0; z < inz; ++z) {
                m = TensorMap(i, j, z, inx, iny);
                w = pesos[TensorMap(m, n, 0, pesosx, pesosy)];
                w -= hitlearn * (tmp * entrada[TensorMap(i, j, z, inx, iny)] + w * decaimentoDePeso);
                pesos[TensorMap(m, n, 0, pesosx, pesosy)] = w;
            }
        }
    }
    oldgrad[n] = tmp;
}

__kernel void fullcalcgrads1(__global double *grad, __global double *gradNext, __global double *input, int dfa,int k0) {
    int n = get_global_id(0) + k0;
    grad[n] = gradNext[n] * func(dfa,input[n]);
}

__kernel void fullcalcgrads2(__global double *grad,__global double * gradsEntrada,__global double * pesos,int pesosx, int pesosy,int k0) {
    int m = get_global_id(0) + k0;
    gradsEntrada[m] = 0;
    for (int n = 0; n < pesosy; ++n) {
        gradsEntrada[m] += grad[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];
    }
}


 long randoml( long seed, long id){
    seed+=id;
    return (seed*0x5deece66dL + 0xbL) & ((1L<<48)-1);
}
double randomD(long seed, long id){
    return(double)randoml(seed,id)/(double) ((1L<<48)-1);
}

__kernel void dropativa(__global double *entrada,__global double *saida,__global char * hitmap, long  seed,double pativa,int k0){
    int i = get_global_id(0) + k0;
    char teste = (char)(randomD(seed,i)<=pativa);
    hitmap[i]= teste;
    saida[i] = teste*entrada[i];
}


 __kernel void dropcalcgrad(__global double *gradentrada,__global char *hitmap,__global double *gradnext,int k0){
    int i = get_global_id(0) + k0;
    gradentrada[i] =  hitmap[i]*gradnext[i];
}
//### guilherme

#endif //CL_TESTE_KERNEL_SRC_H
