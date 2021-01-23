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
    Kernel kernelConvFixWeight;
} *CamadaConv, Typecamadaconv;

void calc_gradsConv(CamadaConv c, Tensor Gradnext);

void releaseConv(CamadaConv *pc);

int ativaConv(CamadaConv c);

void corrige_pesosConv(CamadaConv c);

int convRandomize(CamadaConv c, WrapperCL *cl, GPU_ERROR *error);

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
    if (randomize) convRandomize(c, cl, error);

    c->kernelConvSum = new_Kernel(cl->program, "convSum", 12, VOID_P, VOID_P, VOID_P,
                                  INT, INT, INT, INT, INT, INT, INT, INT, INT);
    c->kernelConFixWeight = new_Kernel(cl->program, "convFixWeight", 8, VOID_P, VOID_P, VOID_P,
                                       DOUBLE, DOUBLE, DOUBLE, DOUBLE, INT);
    return (Camada) c;
}

int convRandomize(CamadaConv c, WrapperCL *cl, GPU_ERROR *error) {
    int lenFilter = c->tamanhoFiltro,
            inz = c->super.entrada->z,
            numeroFiltros = c->numeroFiltros;
    cl_context context = cl->context;
    UINT maxVal = lenFilter * lenFilter * inz;

    double *data = (double *) calloc(lenFilter * lenFilter * inz, sizeof(double));
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, cl->device, NULL, &error->error);
    if (error->error) {
        snprintf(error->msg, 255, "nao foi possivel cruiar a queue\n");
        return error->error;
    }
    for (int a = 0; a < numeroFiltros; a++) {
        FOR3D(i, j, z, lenFilter, lenFilter, inz) {
                    data[TensorMap(c->filtros[a], i, j, z)] = 1.0 / maxVal * (rand() / ((double) RAND_MAX));
                }
        error->error = clEnqueueWriteBuffer(queue, c->filtros[a]->data, CL_TRUE, 0, c->filtros[a]->bytes, data, 0, NULL, NULL);
        if (error->error) {
            snprintf(error->msg, 255, "nao foi possivel copiar dados\n");
            free(data);
            clReleaseCommandQueue(queue);
            return error->error;

        }
    }
    free(data);
    clFinish(queue);
    clReleaseCommandQueue(queue);
    return 0;
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

/***
 * Faz a soma convolutiva  e armazena resultado no tensor saida
 * consulte o kernel convSum em "../kernels/gpu_functions.cl"
 * @param c : objeti camada conv
 * @return : caso diferente de zero ocorreu um erro
 */
int ativaConv(CamadaConv c) {
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

        call_kernel(c->super.saida->x * c->super.saida->y,
                    Kernel_putArgs(&c->kernelConvSum, 12, &filtro->data, &c->super.entrada->data, &c->super.saida->data,
                                   &filtrok, &c->passo, &c->super.saida->x, &c->super.saida->y, &c->super.entrada->x, &c->super.entrada->y, &filtro->x, &filtro->z, &id);
                            error = clEnqueueNDRangeKernel(c->super.queue, c->kernelConvSum.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                            PERR(error, "falha ao chamar kernel convSUm")
        );

        return clFinish(c->super.queue);
    }
}


void corrige_pesosConv(CamadaConv c) {
    double w = 0, m = 0;
    Params *parametros = c->super.parametros;
    Tensor filtro, grad, gradOld;
    int error = 0, id = 0;
    size_t global, local, resto;
    for (int a = 0; a < c->numeroFiltros; a++) {
        filtro = c->filtros[a];
        grad = c->grad_filtros[a];
        gradOld = c->grad_filtros_old[a];
        call_kernel(c->tamanhoFiltro * c->tamanhoFiltro * c->super.entrada->z,
                    Kernel_putArgs(&c->kernelConvFixWeight, 8, &filtro->data, &grad->data, &gradOld->data,
                                   &parametros->hitLearn, &parametros->momento, &parametros->multiplicador, &parametros->decaimentoDePeso, &id);
                            error = clEnqueueNDRangeKernel(c->super.queue, c->kernelConvFixWeight.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                            PERRW(error, "falha ao chamar kernel "));

    }
    clFinish(c->super.queue);
}


void calc_gradsConv(CamadaConv c, Tensor Gradnext) {
    // zerar o gradiente
    double zero =0;
    for (int k = 0; k < c->numeroFiltros; k++) {
        clEnqueueFillBuffer(c->super.queue,c->grad_filtros[k]->data,&zero,sizeof(double),0,c->grad_filtros[k]->bytes,0,NULL,NULL);
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
