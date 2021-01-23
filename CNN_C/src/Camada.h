//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_H
#define CNN_GPU_CAMADA_H
#include"Tensor.h"
// funcao com dois parametros do tipo ponteiro com retorno inteiro
typedef int (*fvv)(void *, void *);

typedef int (*fv)(void *);

typedef struct {
    double hitLearn, momento, decaimentoDePeso;
} Params;

#define CONV 1
#define POOL 2
#define RELU 3
#define DROPOUT 4
#define FULLCONNECT 4
typedef struct {
    fvv calc_grads;
    fv corrige_pesos;
    fv ativa;
    int type;
    fv release;
    Params *parametros;
    Tensor gradsEntrada;
    Tensor entrada;
    Tensor saida;
    char flag_releaseInput;
} *Camada, Typecamada;
#endif //CNN_GPU_CAMADA_H
