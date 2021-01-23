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
    double hitLearn, momento, decaimentoDePeso,multiplicador;
} Params;

#define CONV 1
#define POOL 2
#define RELU 3
#define DROPOUT 4
#define FULLCONNECT 5
typedef struct {
    fvv calc_grads;
    fv corrige_pesos;
    fv ativa;
    fv salvar,carregar;
    int type;
    fv release;
    Params *parametros;
    Tensor gradsEntrada;
    Tensor entrada;
    Tensor saida;
    char flag_releaseInput;

    cl_command_queue queue;
} *Camada, Typecamada;
#endif //CNN_GPU_CAMADA_H
