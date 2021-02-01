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
//#include "../kernels/gpu_functions.cl"

typedef unsigned int UINT;


typedef struct {
    Typecamada super;
    Tensor filtros;
    Tensor grad_filtros;
    Tensor grad_filtros_old;


    UINT passo, tamanhoFiltro, numeroFiltros;

    Kernel kernelConvSum;
    Kernel kernelConvFixWeight;
    Kernel kernelConvCalcGrads;
} *CamadaConv, Typecamadaconv;

void calc_gradsConv(CamadaConv c, Tensor Gradnext);

void releaseConv(CamadaConv *pc);

int ativaConv(CamadaConv c);

void corrige_pesosConv(CamadaConv c);

int convRandomize(CamadaConv c, WrapperCL *cl, GPU_ERROR *error);

void salvarConv(WrapperCL *cl, CamadaConv c, FILE *dst, GPU_ERROR *error);

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
    c->super.salvar = (fsl) salvarConv;
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
    c->filtros = newTensor4D(cl->context, lenFilter, lenFilter, inz, numeroFiltros, error);
    c->grad_filtros = newTensor4D(cl->context, lenFilter, lenFilter, inz, numeroFiltros, error);
    c->grad_filtros_old = newTensor4D(cl->context, lenFilter, lenFilter, inz, numeroFiltros, error);


    if (randomize) convRandomize(c, cl, error);

    c->kernelConvSum = new_Kernel(cl->program, "convSum", 12, VOID_P, VOID_P, VOID_P,
                                  INT, INT, INT, INT, INT, INT, INT, INT, INT);
    c->kernelConvFixWeight = new_Kernel(cl->program, "convFixWeight", 8, VOID_P, VOID_P, VOID_P,
                                        DOUBLE, DOUBLE, DOUBLE, DOUBLE, INT);
    c->kernelConvCalcGrads = new_Kernel(cl->program, "convCalcGrads", 13, VOID_P, VOID_P, VOID_P, VOID_P, VOID_P,
                                        INT, INT, INT, INT, INT, INT, INT, INT);
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
                    data[TensorMap(c->filtros, i, j, z)] = 1.0 / maxVal * (rand() / ((double) RAND_MAX));
                }
        error->error = clEnqueueWriteBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes, data, 0, NULL, NULL);
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
    releaseTensor(&c->filtros);
    releaseTensor(&c->grad_filtros);
    releaseTensor(&c->grad_filtros_old);

    Kernel_release(&c->kernelConvFixWeight);
    Kernel_release(&c->kernelConvSum);
    Kernel_release(&c->kernelConvCalcGrads);
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
    call_kernel(c->super.saida->x * c->super.saida->y * c->numeroFiltros,
                Kernel_putArgs(&c->kernelConvSum, 12, &filtro->data, &c->super.entrada->data, &c->super.saida->data,
                               &c->passo, &c->super.saida->x, &c->super.saida->y, &c->super.entrada->x, &c->super.entrada->y, &filtro->x, &filtro->z, &id);

                        error = clEnqueueNDRangeKernel(c->super.queue, c->kernelConvSum.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERR(error, "falha ao chamar kernel convSUm")
    );

    return clFinish(c->super.queue);

}

void corrige_pesosConv(CamadaConv c) {
    double w = 0, m = 0;
    Params *parametros = c->super.parametros;
    Tensor filtro, grad, gradOld;
    int error = 0, id = 0;
    size_t global, local, resto;
    call_kernel(c->tamanhoFiltro * c->tamanhoFiltro * c->super.entrada->z * c->numeroFiltros,
                Kernel_putArgs(&c->kernelConvFixWeight, 8, &filtro->data, &grad->data, &gradOld->data,
                               &parametros->hitLearn, &parametros->momento, &parametros->multiplicador, &parametros->decaimentoDePeso, &id);
                        error = clEnqueueNDRangeKernel(c->super.queue, c->kernelConvFixWeight.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel fixWeight "));


    clFinish(c->super.queue);
}


void calc_gradsConv(CamadaConv c, Tensor Gradnext) {
    // zerar o gradiente
    char zero = 0;
    clEnqueueFillBuffer(c->super.queue, c->grad_filtros->data, &zero, sizeof(char), 0, c->grad_filtros->bytes, 0, NULL, NULL);

    int error = 0, id = 0;
    size_t global, local, resto;
    call_kernel(c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
                Kernel_putArgs(&c->kernelConvCalcGrads, 13, &c->filtros->data, &c->grad_filtros->data, &c->super.entrada, c->super.gradsEntrada, Gradnext->data, c->tamanhoFiltro,
                               c->passo, c->super.entrada->x, c->super.entrada->y, c->super.saida->x, c->super.saida->y, c->numeroFiltros, id);
                        error = clEnqueueNDRangeKernel(c->super.queue, c->kernelConvCalcGrads.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel calcGrads"));


}

void salvarConv(WrapperCL *cl, CamadaConv c, FILE *dst, GPU_ERROR *error) {
    char flag = '#';
    fwrite(&c->super.type, sizeof(char), 1, dst);
    fwrite(&flag, sizeof(char), 1, dst);
    fwrite(&c->passo, sizeof(UINT), 1, dst);
    fwrite(&c->tamanhoFiltro, sizeof(UINT), 1, dst);
    fwrite(&c->numeroFiltros, sizeof(UINT), 1, dst);
    fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
    fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
    fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
    double *data = callocdouble(c->tamanhoFiltro * c->tamanhoFiltro * c->super.entrada->z);
    cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
    for (int a = 0; a < c->numeroFiltros; a++) {
        clEnqueueReadBuffer(queue, c->filtros->data, CL_TRUE, a*c->filtros->bytes, c->filtros->bytes, data, 0, NULL, NULL);
        fwrite(data, 1, c->filtros->bytes, dst);
    }
    clFinish(queue);
    clReleaseCommandQueue(queue);
}

Camada carregarConv(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error) {
    char flag = 0;
    fread(&flag, sizeof(char), 1, src);
    if (flag != '#')
        fread(&flag, sizeof(char), 1, src);
    UINT passo, tamanhoFiltro, numeroFiltros, inx, iny, inz;
    fread(&passo, sizeof(UINT), 1, src);
    fread(&tamanhoFiltro, sizeof(UINT), 1, src);
    fread(&numeroFiltros, sizeof(UINT), 1, src);
    fread(&inx, sizeof(UINT), 1, src);
    fread(&iny, sizeof(UINT), 1, src);
    fread(&inz, sizeof(UINT), 1, src);
    CamadaConv c = (CamadaConv) createConv(cl, passo, tamanhoFiltro, numeroFiltros, inx, iny, inz, entrada, params, error, 0);
    double *data = callocdouble(c->tamanhoFiltro * c->tamanhoFiltro * c->super.entrada->z);
    cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
    for (int a = 0; a < c->numeroFiltros; a++) {
        fread(data, 1, c->filtros->bytes, src);
        clEnqueueWriteBuffer(queue, c->filtros->data, CL_TRUE, a * c->filtros->bytes, c->filtros->bytes, data, 0, NULL, NULL);
    }
    clFinish(queue);
    clReleaseCommandQueue(queue);
    return (Camada) c;
}

#endif //CNN_GPU_CAMADACONV_H
