//
// Created by Xx220xX on 28/10/2020.
//

#ifndef CNN_GPU_FUNCOESDEATIVACAO_H
#define CNN_GPU_FUNCOESDEATIVACAO_H
#include"math.h"

#define SIGMOIG 0
#define FLAGDIF 1
static double __sigmoid(double x){
    return 1.0/(1+exp(-x));
}
static double __sigmoid_dif(double x){
    double tmp = cosh(x);
    return 1.0/(tmp*tmp);
}
typedef double (*dfd)(double);

dfd funcoesDeAtivacao[2]={__sigmoid,__sigmoid_dif};

#endif //CNN_GPU_FUNCOESDEATIVACAO_H
