#ifndef CNN_GPU_CNN_H
#define CNN_GPU_CNN_H

#include "Camada.h"
#include "CamadaConv.h"
#include "CamadaRelu.h"
#include "CamadaDropOut.h"
#include "CamadaFullConnect.h"
#include "CamadaPool.h"

#ifdef LOG_CNN_ADD_LAYERS
#undef LOG_CNN_ADD_LAYERS
#define LOG_CNN_ADD_LAYERS(format, ...) printf(format,## __VA_ARGS__);printf("\n");
#else
#define LOG_CNN_ADD_LAYERS(format, ...)
#endif


#define INVALID_FILTER_SIZE (-1)
#define CNN_FLAG_CALCULE_ERROR 1
#define CNN_FLAG_CALCULE_MAX 2
typedef struct _cnn {
    Params parametros;
    Camada *camadas;
    int size;
    Ponto3d sizeIn;
    char err;
    cl_command_queue queue;
    WrapperCL *cl;
    char releaseCL;
    GPU_ERROR error;
    Kernel kernelsub;
    Kernel kerneldiv;
    Kernel kerneldivInt;
    Kernel kernelNorm;
    Kernel kernelMax;
    Kernel kernelInt2Vector;
    char flags;
    double normaErro;
    double indiceSaida;

} *Cnn, TypeCnn;

Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz) {
    Cnn c = (Cnn) calloc(1, sizeof(TypeCnn));
    snprintf(c->error.msg,255,"");
    c->parametros = p;
    c->sizeIn = (Ponto3d) {inx, iny, inz};
    c->cl = cl;
    int error = 0;
    c->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error);
    if(error){
        c->error.error = error;
        snprintf(c->error.msg,255,"nao foi possivel criar queue\n");
    }
    c->kernelsub = new_Kernel(cl->program, "sub", 4, VOID_P, VOID_P, VOID_P, INT);
    c->kerneldiv= new_Kernel(cl->program, "div", 3, VOID_P, DOUBLE,  INT);
    c->kerneldivInt = new_Kernel(cl->program, "divIntDo", 4, VOID_P, VOID_P, DOUBLE, INT);
    c->kernelNorm = new_Kernel(cl->program, "norm", 3, VOID_P, VOID_P, INT);
    c->kernelMax = new_Kernel(cl->program, "maxID", 3, VOID_P, VOID_P, INT);
    c->kernelInt2Vector = new_Kernel(cl->program, "int2vector", 4, VOID_P, VOID_P, INT,INT);
    setmaxWorks(cl->maxworks);
    return c;
}

Cnn createCnnWithgpu(char *kernelFile, Params p, UINT inx, UINT iny, UINT inz) {
    WrapperCL *cl = (WrapperCL *) calloc(sizeof(WrapperCL), 1);
    WrapperCL_initbyFile(cl, kernelFile);
    Cnn c = createCnn(cl, p, inx, iny, inz);
    c->releaseCL = 1;
    return c;
}

Ponto3d __addLayer(Cnn c) {
    c->size += 1;
    c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
    Ponto3d in = c->sizeIn;
    if (c->size > 1) {
        in.x = (int)c->camadas[c->size - 2]->saida->x;
        in.y = (int)c->camadas[c->size - 2]->saida->y;
        in.z = (int)c->camadas[c->size - 2]->saida->z;
    }
    return in;
}

#define checkSizeFilter(v, tam, pas) (((v)-(tam))/(pas)) ==((double)(v)-(tam))/((double)(pas))

int CnnAddConvLayer(Cnn c, UINT passo, UINT tamanhoDoFiltro, UINT numeroDeFiltros) {
    /** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
    * em que:
    * 			S = tamanho da saida
    * 			E = tamanho da entrada
    * 			F = tamanho do filtro
    * 			Pd = preenchimento com zeros
    * 			P = passo
    **/

    Ponto3d sizeIn = __addLayer(c);
    if (!checkSizeFilter(sizeIn.x, tamanhoDoFiltro, passo) || !checkSizeFilter(sizeIn.y, tamanhoDoFiltro, passo)) {
        c->err = INVALID_FILTER_SIZE;
        c->size--;
        c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
        fprintf(stderr, "tamanho do filtro invalido\n");
        return c->err;

    }

    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createConv(c->cl, passo, tamanhoDoFiltro, numeroDeFiltros, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
                                         &c->parametros, &c->error, 1);
    c->camadas[c->size - 1]->queue = c->queue;
    if (!c->error.error) {
        LOG_CNN_ADD_LAYERS("camada convolutiva adicionada");
        LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y, c->camadas[c->size - 1]->saida->z);
    }
    return c->error.error;
}

int CnnAddPoolLayer(Cnn c, UINT passo, UINT tamanhoDoFiltro) {
    /** o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
    * em que:
    * 			S = tamanho da saida
    * 			E = tamanho da entrada
    * 			F = tamanho do filtro
    * 			Pd = preenchimento com zeros
    * 			P = passo
    **/

    Ponto3d sizeIn = __addLayer(c);
    if (!checkSizeFilter(sizeIn.x, tamanhoDoFiltro, passo) || !checkSizeFilter(sizeIn.y, tamanhoDoFiltro, passo)) {
        c->err = INVALID_FILTER_SIZE;
        c->size--;
        c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
        return c->err;

    }

    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createPool(c->cl, passo, tamanhoDoFiltro, sizeIn.x, sizeIn.y, sizeIn.z, entrada, &c->parametros, &c->error);
    c->camadas[c->size - 1]->queue = c->queue;
    if (!c->error.error) {
        LOG_CNN_ADD_LAYERS("camada pooling adicionada");
        LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y, c->camadas[c->size - 1]->saida->z,);
    }
    return c->error.error;

}

int CnnAddReluLayer(Cnn c) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createRelu(c->cl, sizeIn.x, sizeIn.y, sizeIn.z, entrada, &c->error);
    c->camadas[c->size - 1]->queue = c->queue;
    if (!c->error.error) {

        LOG_CNN_ADD_LAYERS("camada relu adicionada");
        LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y, c->camadas[c->size - 1]->saida->z);
    }
    return c->error.error;

}

int CnnAddDropOutLayer(Cnn c, double pontoAtivacao, long long int seed) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createDropOut(c->cl, sizeIn.x, sizeIn.y, sizeIn.z, pontoAtivacao, seed, entrada, &c->error);
    c->camadas[c->size - 1]->queue = c->queue;
    if (!c->error.error) {
        LOG_CNN_ADD_LAYERS("camada dropout adicionada");
        LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y, c->camadas[c->size - 1]->saida->z);
    }
    return c->error.error;
}

int CnnAddFullConnectLayer(Cnn c, UINT tamanhoDaSaida, int funcaoDeAtivacao) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createFullConnect(c->cl, sizeIn.x, sizeIn.y, sizeIn.z, tamanhoDaSaida, entrada, &c->parametros, funcaoDeAtivacao, 1, &c->error);
    c->camadas[c->size - 1]->queue = c->queue;
    if (!c->error.error) {
        LOG_CNN_ADD_LAYERS("camada full connect adicionada");
        LOG_CNN_ADD_LAYERS("SAIDA(%d,%d,%d)", c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y, c->camadas[c->size - 1]->saida->z);
    }
    return c->error.error;
}

int CnnCall(Cnn c, double *input) {
    c->error.error = clEnqueueWriteBuffer(c->queue, c->camadas[0]->entrada->data, CL_TRUE, 0, c->camadas[0]->entrada->bytes, input, 0, NULL, NULL);
    for (int i = 0; i < c->size; ++i) {
        c->camadas[i]->ativa(c->camadas[i]);
    }
    size_t global = 1,local = 1;
    if(c->flags&CNN_FLAG_CALCULE_MAX){
        Tensor saida = c->camadas[c->size-1]->saida;
        Tensor entrada = c->camadas[0]->entrada;
        int len = saida->x*saida->y*saida->z;
        Kernel_putArgs(&c->kernelMax, 3, &saida->data,&entrada->data,&len);
        int error = clEnqueueNDRangeKernel(c->queue, c->kernelMax.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        PERRW(error, "falha ao chamar kernel norm")
        clEnqueueReadBuffer(c->queue,entrada->data,CL_TRUE,0,sizeof(double),&c->indiceSaida,0,NULL,NULL);
    }
    return c->error.error;
}

int CnnLearn(Cnn c, double *target) {
    if (c->size == 0)return -1;
    Tensor lastGrad, targ;
    Tensor gradNext;
    lastGrad = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y, c->camadas[c->size - 1]->saida->z, &c->error);
    targ = newTensor(c->cl->context, c->camadas[c->size - 1]->saida->x, c->camadas[c->size - 1]->saida->y, c->camadas[c->size - 1]->saida->z, &c->error);
    clEnqueueWriteBuffer(c->queue, targ->data, CL_TRUE, 0, targ->bytes, target, 0, NULL, NULL);

    int error = 0, id = 0;
    size_t global, local, resto;
    LOG_CNN_KERNELCALL("Chamando kernel sub")
    call_kernel(targ->x * targ->y * targ->z,
                Kernel_putArgs(&c->kernelsub, 4, &lastGrad->data, &c->camadas[c->size - 1]->saida->data, &targ->data, &id);
                        error = clEnqueueNDRangeKernel(c->queue, c->kernelsub.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel sub")
    );

    gradNext = lastGrad;
    for (int l = c->size - 1; l >= 0; l--) {
        c->camadas[l]->calc_grads(c->camadas[l], gradNext);
        if (!c->camadas[l]->flag_notlearn)
            c->camadas[l]->corrige_pesos(c->camadas[l]);
        gradNext = c->camadas[l]->gradsEntrada;

    }
    if(c->flags&CNN_FLAG_CALCULE_ERROR){
        global = 1;
        local = 1;
        int len = lastGrad->x*lastGrad->y*lastGrad->z;
        Kernel_putArgs(&c->kernelNorm, 3, &lastGrad->data,&targ->data,&len);
        error = clEnqueueNDRangeKernel(c->queue, c->kernelNorm.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        PERRW(error, "falha ao chamar kernel norm")
        clEnqueueReadBuffer(c->queue,targ->data,CL_TRUE,0,sizeof(double),&c->normaErro,0,NULL,NULL);
    }
    releaseTensor(&lastGrad);
    releaseTensor(&targ);
}

void releaseCnn(Cnn *pc) {
    Cnn c = *pc;
    if(!c)return;
    for (int i = 0; i < c->size; ++i) {
        c->camadas[i]->release(c->camadas + i);
    }
    free(c->camadas);
    clReleaseCommandQueue(c->queue);
    Kernel_release(&c->kernelsub);
    Kernel_release(&c->kerneldiv);
    Kernel_release(&c->kerneldivInt);
    Kernel_release(&c->kernelNorm);
    Kernel_release(&c->kernelMax);
    if (c->releaseCL) {
        WrapperCL_release(c->cl);
        free(c->cl);
    }
    free(c);
    *pc = NULL;
}

void cnnSave(Cnn c, FILE *dst) {
    int i;
    for ( i = 0; i < c->size; ++i) {
        c->camadas[i]->salvar(c->cl, c->camadas[i], dst, &c->error);
        if (c->error.error < 0)break;
    }
    if (i!=c->size){
        if(!c->error.error){
            c->error.error = -10;
            snprintf(c->error.msg,255,"falha ao salvar camadas\n");
        }
    }
}

int cnnCarregar(Cnn c, FILE *src) {
    if (c->size != 0)return -1;
    Camada cm;
    Tensor entrada = NULL;
    while (1) {
        cm = carregarCamada(c->cl, src, entrada, &c->parametros, &c->error);
        if (cm == NULL) { break; }
        entrada = cm->saida;
        __addLayer(c);
        c->camadas[c->size - 1] = cm;
        cm->queue = c->queue;
        if (c->error.error < 0)break;
    }
    if (c->size > 0) {
        c->sizeIn.x = (int) c->camadas[0]->entrada->x;
        c->sizeIn.y = (int) c->camadas[0]->entrada->y;
        c->sizeIn.z = (int) c->camadas[0]->entrada->z;
    }

    return c->error.error;
}
void Cnngetout(Cnn c, double *out){
    if(c->size<1)return;
    clEnqueueReadBuffer(c->queue,c->camadas[c->size-1]->saida->data,CL_TRUE,0,c->camadas[c->size-1]->saida->bytes,out,0,NULL,NULL);
}


#endif