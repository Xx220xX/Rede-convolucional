//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADADROPOUT_H
#define CNN_GPU_CAMADADROPOUT_H

#include "string.h"
#include"Camada.h"
#include"Tensor.h"
#include <stdlib.h>
#include <float.h>



typedef struct {
    Typecamada super;
    TensorChar hitmap;
    char flag_releaseInput;
    double p_ativacao;
    cl_long seed;
    Kernel kerneldropativa;
    Kernel kerneldropcalcgrad;
} *CamadaDropOut, Typecamadadropout;

void releaseDropOut(CamadaDropOut *pc);

void corrigePesosDropOut(CamadaDropOut c);

void ativaDropOut(CamadaDropOut c);

void calc_gradsDropOut(CamadaDropOut c, Tensor GradNext);
void salvarDropOut(WrapperCL *cl, CamadaDropOut c, FILE *dst, GPU_ERROR *error);
Camada createDropOut(WrapperCL *cl,UINT inx, UINT iny, UINT inz, double p_ativacao,long long seed, Tensor entrada,GPU_ERROR *error) {
    if (error->error)return NULL;
    CamadaDropOut c = (CamadaDropOut) calloc(1, sizeof(Typecamadadropout));
    c->super.gradsEntrada = newTensor(cl->context,inx, iny, inz,error);
    if (!entrada) {
        c->super.entrada = newTensor(cl->context,inx, iny, inz,error);
        c->flag_releaseInput = 1;
    } else {
        c->super.entrada = entrada;
    }
    c->super.saida = newTensor(cl->context,inx, iny, inz,error);
    c->hitmap = newTensorChar(cl->context,inx, iny, inz,error);
    c->p_ativacao = p_ativacao;
    c->super.release = (fv) releaseDropOut;
    c->super.ativa = (fv)ativaDropOut;
    c->super.calc_grads =(fvv) calc_gradsDropOut;
    c->super.corrige_pesos = (fv)corrigePesosDropOut;
    c->super.type = DROPOUT;
    c->seed = seed;
    c->super.salvar =(fsl) salvarDropOut;
    c->kerneldropativa = new_Kernel(cl->program, "dropativa", 6, VOID_P, VOID_P, VOID_P,sizeof(cl_long),DOUBLE,INT);
    c->kerneldropcalcgrad = new_Kernel(cl->program, "dropcalcgrad", 4, VOID_P, VOID_P, VOID_P,INT);
    return (Camada)c;
}

void releaseDropOut(CamadaDropOut *pc) {
    CamadaDropOut c = *pc;
    releaseTensor(&c->super.gradsEntrada);
    releaseTensor(&c->super.saida);
    releaseTensorChar(&c->hitmap);
    if (c->flag_releaseInput)releaseTensor(&c->super.entrada);
    Kernel_release(&c->kerneldropcalcgrad);
    Kernel_release(&c->kerneldropativa);
    free(c);
    *pc = 0;
}

void ativaDropOut(CamadaDropOut c) {
    int error = 0, id = 0;
    size_t global, local, resto;
    LOG_CNN_KERNELCALL("ativa drop: ativadrop")
    call_kernel(c->super.saida->x*c->super.saida->y*c->super.saida->z,
                Kernel_putArgs(&c->kerneldropativa, 6,&c->super.entrada->data,&c->super.saida->data,&c->hitmap->data,&c->seed,&c->p_ativacao
                      , &id);
                        error = clEnqueueNDRangeKernel(c->super.queue, c->kerneldropativa.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel ativa dropout")
    );
    c->seed +=c->super.saida->x*c->super.saida->y*c->super.saida->z;
    c->seed = (c->seed*0x5deece66dLL + 0xbLL) & ((1LL<<48)-1);
}


void corrigePesosDropOut(CamadaDropOut c) {}

void calc_gradsDropOut(CamadaDropOut c, Tensor GradNext) {
    int error = 0, id = 0;
    size_t global, local, resto;
    LOG_CNN_KERNELCALL("calc drop: calcgrad")
    call_kernel(c->super.saida->x*c->super.saida->y*c->super.saida->z,
                error = kernel_run(&c->kerneldropcalcgrad,c->super.queue,global,local,&c->super.gradsEntrada->data,&c->hitmap->data,&GradNext->data, &id);
                         PERRW(error, "falha ao chamar kernel cacalcgrad dropout")
    );
}
void salvarDropOut(WrapperCL *cl, CamadaDropOut c, FILE *dst, GPU_ERROR *error) {
    LOG_CNN_SALVE_LAYERS("salvando drop out")
    char flag = '#';
    fwrite(&c->super.type, sizeof(char), 1, dst);
    fwrite(&flag, sizeof(char), 1, dst);
    fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
    fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
    fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
    fwrite(&c->p_ativacao, sizeof(double), 1, dst);
    LOG_CNN_SALVE_LAYERS("salvou com erro %d: %s",error->error,error->msg)

}

Camada carregarDropOut(WrapperCL *cl, FILE *src, Tensor entrada,Params *params,GPU_ERROR *error) {
    if (error->error)return NULL;
    char flag = 0;
    fread(&flag, sizeof(char), 1, src);
    if (flag != '#')
        fread(&flag, sizeof(char), 1, src);
    UINT inx, iny, inz;
    double pativacao;
    fread(&inx, sizeof(UINT), 1, src);
    fread(&iny, sizeof(UINT), 1, src);
    fread(&inz, sizeof(UINT), 1, src);
    fread(&pativacao, sizeof(double), 1, src);
    CamadaDropOut c = (CamadaDropOut)createDropOut(cl,inx,iny,inz,pativacao,time(NULL),entrada,error);
    return (Camada) c;
}

#endif //CNN_GPU_CAMADADROPOUT_H
