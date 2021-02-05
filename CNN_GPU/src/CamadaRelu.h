//
// Created by gf154 on 24/10/2020.
//

#ifndef CNN_GPU_CAMADA_RELU_H
#define CNN_GPU_CAMADA_RELU_H

#include "Camada.h"
#include"Tensor.h"
#include <stdlib.h>

typedef struct {
    Typecamada super;
    Kernel kernelReluAtiva;
    Kernel kernelReluCalcGrads;
} *CamadaRelu, TypecamadaRelu;

void realeaseRelu(CamadaRelu *pc);

void ativaRelu(CamadaRelu c);

void corrige_pesosRelu(CamadaRelu);

void calc_gradsRelu(CamadaRelu c, Tensor GradNext);

void salvarRelu(WrapperCL *cl, CamadaRelu c, FILE *dst, GPU_ERROR *error);

Camada createRelu(WrapperCL *cl, unsigned int inx, unsigned int iny, unsigned int inz, Tensor entrada, GPU_ERROR *error) {
    CamadaRelu c = (CamadaRelu) calloc(1, sizeof(TypecamadaRelu));

    c->super.gradsEntrada = newTensor(cl->context, inx, iny, inz, error);
    c->super.saida = newTensor(cl->context, inx, iny, inz, error);
    c->super.entrada = entrada;
    if (!entrada) {
        c->super.entrada = newTensor(cl->context, inx, iny, inz, error);
        c->super.flag_releaseInput = 1;
    }
    c->super.release = (fv) realeaseRelu;
    c->super.ativa = (fv) ativaRelu;
    c->super.calc_grads = (fvv) calc_gradsRelu;
    c->super.corrige_pesos = (fv) corrige_pesosRelu;
    c->super.type = RELU;
    c->super.salvar = (fsl) salvarRelu;

    c->kernelReluAtiva = new_Kernel(cl->program, "reluativa", 3, VOID_P, VOID_P, INT);
    c->kernelReluCalcGrads = new_Kernel(cl->program, "relucalcgrad", 4, VOID_P, VOID_P, VOID_P, INT);
    return (Camada) c;
}

void realeaseRelu(CamadaRelu *pc) {
    CamadaRelu c = *pc;
    releaseTensor(&c->super.gradsEntrada);
    if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
    releaseTensor(&c->super.saida);
    Kernel_release(&c->kernelReluCalcGrads);
    Kernel_release(&c->kernelReluAtiva);
    free(c);
    *pc = NULL;
}

void ativaRelu(CamadaRelu c) {
    int error = 0, id = 0;
    size_t global, local, resto;
    call_kernel(c->super.saida->x * c->super.saida->y * c->super.saida->z,
                Kernel_putArgs(&c->kernelReluAtiva, 3, &c->super.entrada->data, &c->super.saida->data, &id);
                        error = clEnqueueNDRangeKernel(c->super.queue, c->kernelReluAtiva.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel ativa relu")
    );
}

void corrige_pesosRelu(CamadaRelu c) {}

void calc_gradsRelu(CamadaRelu c, Tensor GradNext) {
    int error = 0, id = 0;
    size_t global, local, resto;
    call_kernel(c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
                Kernel_putArgs(&c->kernelReluCalcGrads, 4, &c->super.gradsEntrada->data, &c->super.entrada->data, &GradNext->data, &id);
                        error = clEnqueueNDRangeKernel(c->super.queue, c->kernelReluCalcGrads.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel ativa dropout")
    );
}

void salvarRelu(WrapperCL *cl, CamadaRelu c, FILE *dst, GPU_ERROR *error) {
    char flag = '#';
    fwrite(&c->super.type, sizeof(char), 1, dst);
    fwrite(&flag, sizeof(char), 1, dst);
    fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
    fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
    fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
}

Camada carregarRelu(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error) {
    char flag = 0;
    fread(&flag, sizeof(char), 1, src);
    if (flag != '#')
        fread(&flag, sizeof(char), 1, src);
    UINT passo, tamanhoFiltro, inx, iny, inz;
    fread(&inx, sizeof(UINT), 1, src);
    fread(&iny, sizeof(UINT), 1, src);
    fread(&inz, sizeof(UINT), 1, src);
    return createRelu(cl, inx, iny, inz, entrada, error);
}

#endif //CNN_GPU_CAMADA_RELU_H
