//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_H
#define CNN_GPU_CAMADA_H

#include"Tensor.h"
#include <time.h>
typedef struct {
    double hitLearn, momento, decaimentoDePeso, multiplicador;
} Params;
// funcao com dois parametros do tipo ponteiro com retorno inteiro

typedef int (*fvv)(void *, void *);

typedef int (*fv)(void *);

typedef void  (*fsl)(void *, void *, void *, void *);


#define CONV 1
#define POOL 2
#define RELU 3
#define DROPOUT 4
#define FULLCONNECT 5

typedef struct {
    fvv calc_grads;
    fv corrige_pesos;
    fv ativa;
    fsl salvar;
    char type;
    fv release;
    Params *parametros;
    Tensor gradsEntrada;
    Tensor entrada;
    Tensor saida;
    char flag_releaseInput;
    char flag_notlearn;
    cl_command_queue queue;
} *Camada, Typecamada;
size_t  max_works=1;
void setmaxWorks(size_t max_){
    max_works = max_;
}
Camada carregarConv(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarPool(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarRelu(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarDropOut(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarFullConnect(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error);

Camada carregarCamada(WrapperCL *cl, FILE *src, Tensor entrada, Params *param, GPU_ERROR *error) {
    char identify = 0;
    fread(&identify, sizeof(char), 1, src);
    if (feof(src))return NULL;
    switch (identify) {
        case CONV:
            return carregarConv(cl, src, entrada, param, error);
        case POOL:
            return carregarPool(cl, src, entrada, param, error);
        case RELU:
            return carregarRelu(cl, src, entrada, param, error);
        case DROPOUT:
            return carregarDropOut(cl, src, entrada, param, error);
        case FULLCONNECT:
            return carregarFullConnect(cl, src, entrada, param, error);
        default:
            return NULL;
    }

}

#ifdef LOG_CNN_SALVE_LAYERS
#undef LOG_CNN_SALVE_LAYERS
#define LOG_CNN_SALVE_LAYERS(format, ...) printf(format,## __VA_ARGS__);printf("\n");
#else
#define LOG_CNN_SALVE_LAYERS(format, ...)
#endif

#ifdef LOG_CNN_KERNELCALL
#undef LOG_CNN_KERNELCALL
#define LOG_CNN_KERNELCALL(format, ...) printf(format,## __VA_ARGS__);printf("\n");
#else
#define LOG_CNN_KERNELCALL(format, ...)
#endif

#endif //CNN_GPU_CAMADA_H
