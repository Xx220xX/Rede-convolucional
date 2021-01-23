//
// Created by Xx220xX on 10/05/2020.
//a
#ifndef CL_KERNEL_SRC_H
#define CL_KERNEL_SRC_H

typedef double (*dfd)(double);

double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }

double difsigmoid(double x) {
    double tmp = sigmoid(x);
    return tmp * (1 - tmp);
}

double tanghG(double x) { return tanh(x); }

double diftanhG(double x) {
    double tmp = tanh(x);
    return (1 - tmp * tmp);
}

double relu(double x) { return x > 0 ? x : 0.0; }

double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }

dfd func[6] = {sigmoid, difsigmoid, tanghG, diftanhG, relu, difrelu};

void initFucntions() {
    func[0] = (void *) sigmoid;
    func[1] = (void *) difsigmoid;
    func[2] = (void *) tanghG;
    func[3] = (void *) diftanhG;
    func[4] = (void *) relu;
    func[5] = (void *) difrelu;
}
#define TensorMap(x,y,z,tx,ty)(z)*(ty*tx)+(y)*tx+(x)
typedef struct {
    int x, y, z;
} Ponto3d;

__kernel void convSum(__global double *filtro, __global double *entrada, __global double *saida,
                      int filtrok, int passo, int saidatx, int saidaty, int entradatx, int entradaty,
                      int lenFilter, int filtroz, int k0) {
    int k = get_global_id(0) + k0;
    y = k % saidaty;
    x = k / saidaty;
    Ponto3d mapeado = {x * passo, y * passo, 0};
    double sum = 0,f,v;
    for (int i = 0; i < lenFilter; i++)
        for (int j = 0; j < lenFilter; j++)
            for (int z = 0; z < filtroz; z++) {
                f = filtro[TensorMap(i,j,z,lenFilter,lenFilter)];
                v = entrada[TensorMap(mapeado.x+i,mapeado.y+j,z,entradatx,entradaty)];
                sum+= f*v;
            }
    saida[TensorMap(x,y,filtrok,saidatx,saidaty)] = sum;
}
__kernel void convFixWeight(__global double *filtro,__global double *grad,__global double *gradOld,double hitlearn,
                            double momento,double multp,double weightDecay,int k0){
    int k = get_global_id(0) + k0;
    double m  = grad[k] + gradOld[k]*momento;
    double w = filtro[k] ;
    filtro[k] =   w - hitlearn * (m * multp +w * weightDecay);
    gradOld[k] = m;

}

#endif //CL_TESTE_KERNEL_SRC_H
