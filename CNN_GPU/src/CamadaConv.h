//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADACONV_H
#define CNN_GPU_CAMADACONV_H

#include "string.h"
#include"Camada.h"
#include"Tensor.h"
#include <stdlib.h>
#include"utils.h"

typedef unsigned int UINT;


typedef struct {
    Typecamada super;
    Tensor *filtros;
    Tensor *grad_filtros;
    Tensor *grad_filtros_old;
    UINT passo, tamanhoFiltro, numeroFiltros;

    Kernel kernelConvSum;
} *CamadaConv, Typecamadaconv;

void calc_gradsConv(CamadaConv c, Tensor Gradnext);

void releaseConv(CamadaConv *pc);

void ativaConv(CamadaConv c);

void corrige_pesosConv(CamadaConv c);

Camada createConv(WrapperCL *cl, UINT passo, UINT lenFilter, UINT numeroFiltros, UINT inx, UINT iny, UINT inz,
                  Tensor entrada, Params *params, GPU_ERROR *error, int randomize) {
    CamadaConv c = (CamadaConv) calloc(1, sizeof(Typecamadaconv));
    cl_context context = cl->context;
    c->super.release = (fv) releaseConv;
    c->super.ativa = (fv) ativaConv;
    c->super.calc_grads = (fvv) calc_gradsConv;
    c->super.corrige_pesos = (fv) corrige_pesosConv;
    c->super.parametros = params;
    c->super.type = CONV;

    c->passo = passo;
    c->tamanhoFiltro = lenFilter;
    c->numeroFiltros = numeroFiltros;
    c->super.gradsEntrada = newTensor(context, inx, iny, inz, error);
    if (error->error)return (Camada) c;

    c->super.entrada = entrada;
    if (!entrada) {
        c->super.entrada = newTensor(context, inx, iny, inz, error);
        c->super.flag_releaseInput = 1;

    }
    c->super.saida = newTensor(context, (inx - lenFilter) / passo + 1, (iny - lenFilter) / passo + 1, numeroFiltros, error);
    if (error->error)return (Camada) c;
    c->filtros = (Tensor *) calloc(numeroFiltros, sizeof(Tensor));
    c->grad_filtros = (Tensor *) calloc(numeroFiltros, sizeof(Tensor));
    c->grad_filtros_old = (Tensor *) calloc(numeroFiltros, sizeof(Tensor));
    Tensor tmp;
    UINT maxVal = lenFilter * lenFilter * inz;
    for (int a = 0; a < numeroFiltros; a++) {
        tmp = newTensor(context, lenFilter, lenFilter, inz, error);
        if (error->error) {
            c->numeroFiltros = a;
            return (Camada) c;
        }
        c->filtros[a] = tmp;
        c->grad_filtros[a] = newTensor(context, lenFilter, lenFilter, inz, error);
        if (error->error) {
            return (Camada) c;
        }
    }
    if (randomize) {
        double *data = (double *) calloc(lenFilter * lenFilter * inz, sizeof(double));
        cl_command_queue queue = clCreateCommandQueueWithProperties(context, cl->device, NULL, &error->error);
        if(error->error){
            snprintf(error->msg,255,"nao foi possivel cruiar a queue\n");
            return (Camada)c;
        }
        for (int a = 0; a < numeroFiltros; a++) {
            FOR3D(i, j, z, lenFilter, lenFilter, inz) {
                        data[TensorMap(tmp, i, j, z)] = 1.0 / maxVal * (rand() / ((double) RAND_MAX));
                    }
            error->error= clEnqueueWriteBuffer(queue,c->filtros[a]->data,CL_TRUE,0,c->filtros[a]->bytes,data,0,NULL,NULL);
            if(error->error){
                snprintf(error->msg,255,"nao foi possivel copiar dados\n");
                free(data);
                clReleaseCommandQueue(queue);
                return (Camada)c;
            }
        }
        free(data);
        clReleaseCommandQueue(queue);

    }
    return (Camada) c;
}

void releaseConv(CamadaConv *pc) {
    CamadaConv c = *pc;
    releaseTensor(&c->super.gradsEntrada);
    if (c->super.flag_releaseInput)
        releaseTensor(&c->super.entrada);
    releaseTensor(&c->super.saida);
    for (int a = 0; a < c->numeroFiltros; a++) {
        releaseTensor(c->filtros + a);
        releaseTensor(c->grad_filtros + a);
        releaseTensor(c->grad_filtros_old + a);
    }
    free(c->filtros);
    free(c);
    *pc = NULL;
}

void ativaConv(CamadaConv c) {
    Tensor filtro;
    Ponto3d mapeado;
    double sum, f, v;
    Tensor entrada = c->super.entrada;
    //iteraçao nos filtros
    int error = 0, id = 0;
    size_t global, local, resto;

    for (int filtrok = 0; filtrok < c->numeroFiltros; filtrok++) {
        //seleciona o filtro
        filtro = c->filtros[filtrok];

        call_kernel(c->super.saida->x*c->super.saida->y,
                    Kernel_putArgs(&c->kernelConvSum, 12, &filtro->data, &c->super.entrada->data, &c->super.saida->data,
                                   &filtrok, &c->passo,&c->super.saida->x,&c->super.saida->y,&c->super.entrada->x,&c->super.entrada->y,&filtro->x,&filtro->z, &id);
                            error = clEnqueueNDRangeKernel(c->super.queue, c->kernelConvSum.kernel, 1, NULL, &global, &local, 0, NULL,NULL);
                           PERR(error, "falha ao chamar kernel AL-Y")
        );


    }
}


void corrige_pesosConv(CamadaConv c) {
    double w = 0, m = 0;
    double grad, oldGrad;
    Params *parametros = c->super.parametros;
    for (int a = 0; a < c->numeroFiltros; a++) {
        FOR3D(i, j, z, c->tamanhoFiltro, c->tamanhoFiltro, c->super.entrada->tz) {
                    w = TensorAT(c->filtros[a], i, j, z);
                    grad = TensorAT(c->grad_filtros[a], i, j, z);
                    oldGrad = TensorAT(c->grad_filtros_old[a], i, j, z);
                    m = grad + oldGrad * parametros->momento;
                    TensorAT(c->filtros[a], i, j, z) =
                            w - parametros->hitLearn * (m * TensorAT(c->super.entrada, i, j, z) +
                                                        w * parametros->decaimentoDePeso);
                    TensorAT(c->grad_filtros_old[a], i, j, z) =
                            grad + oldGrad * parametros->momento;


                }
    }
}


void calc_gradsConv(CamadaConv c, Tensor Gradnext) {
    // zerar o gradiente
    for (int k = 0; k < c->numeroFiltros; k++) {
        memset((void *) c->grad_filtros[k]->data, 0, c->tamanhoFiltro * c->tamanhoFiltro * c->grad_filtros[k]->tz);
    }
    Range range = {0};
    double somaErro = 0;
    int minX, minY;
    double pesoAplicado;
    FOR2D(x, y, c->super.entrada->tx, c->super.entrada->ty) {
            range = mapeia_entrada_saida(x, y, c->passo, c->tamanhoFiltro, c->super.saida, c->numeroFiltros);
            printPonto3D(range.min);
            printPonto3D(range.max);
            printf("\n");
            for (int z = 0; z < c->super.entrada->tz; z++) {
                somaErro = 0;
                for (int i = range.min.x; i <= range.max.x; i++) {
                    minX = i * c->passo;
                    for (int j = range.min.y; j <= range.max.y; j++) {
                        minY = j * c->passo;
                        for (int k = range.min.z; k <= range.max.z; k++) {
                            pesoAplicado = TensorAT(c->filtros[k], x - minX, y - minY, z);
                            somaErro += pesoAplicado * TensorAT(Gradnext, i, j, k);
                            TensorAT(c->grad_filtros[k], x - minX, y - minY, z) +=
                                    TensorAT(c->super.entrada, x, y, z) * TensorAT(Gradnext, i, j, k);
                        }
                    }
                }
                TensorAT(c->super.gradsEntrada, x, y, z) = somaErro;

            }
        }
}

#endif //CNN_GPU_CAMADACONV_H
