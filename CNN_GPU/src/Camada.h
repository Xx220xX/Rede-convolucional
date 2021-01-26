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
    double hitLearn, momento, decaimentoDePeso, multiplicador;
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
    fv salvar;
    char type;
    fv release;
    Params *parametros;
    Tensor gradsEntrada;
    Tensor entrada;
    Tensor saida;
    char flag_releaseInput;

    cl_command_queue queue;
} *Camada, Typecamada;
Camada carregarConv(WrapperCL *cl, FILE *src, Tensor entrada,Params *params,GPU_ERROR *error);
Camada carregarPool(WrapperCL *cl, FILE *src, Tensor entrada,Params *params,GPU_ERROR *error);
Camada carregarRelu(WrapperCL *cl, FILE *src, Tensor entrada,Params *params,GPU_ERROR *error);
Camada carregarDropOut(WrapperCL *cl, FILE *src, Tensor entrada,Params *params,GPU_ERROR *error);
Camada carregarFullConnect(WrapperCL *cl, FILE *src, Tensor entrada,Params *params,GPU_ERROR *error);

Camada carregarCamada(WrapperCL *cl,FILE * src,GPU_ERROR *error) {
    char identify =0;
    fread(&identify,sizeof(char),1,src);
    if (feof(src))return NULL;
    switch (identify) {
        case CONV:
            return carregarConv(cl,src,error);
        case POOL:
            return carregarPool(cl,src,error);
        case RELU:
            return carregarRelu(cl,src,error);
        case DROPOUT:
            return carregarDropOut(cl,src,error);
        case FULLCONNECT:
            return carregarFullConnect(cl,src,error);
        default:return NULL;
    }

}

#endif //CNN_GPU_CAMADA_H
