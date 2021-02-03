#ifndef CNN_GPU_CNN_H
#define CNN_GPU_CNN_H

#include "Camada.h"
#include "CamadaConv.h"
#include "CamadaRelu.h"
#include "CamadaDropOut.h"
#include "CamadaFullConnect.h"
#include "CamadaPool.h"


#define INVALID_FILTER_SIZE (-1)

typedef struct _cnn {
    Params parametros;
    Camada *camadas;
    int size;
    Ponto3d sizeIn;
    char err;
    cl_command_queue queue;
    WrapperCL *cl;
    GPU_ERROR  error;
    Kernel kernelsub;
} *Cnn, TypeCnn;

Cnn createCnn(WrapperCL  *cl,Params p, UINT inx, UINT iny, UINT inz) {
    Cnn c = (Cnn) calloc(1, sizeof(TypeCnn));
    c->parametros = p;
    c->sizeIn = (Ponto3d) {inx, iny, inz};
    c->cl = cl;
    int error = 0;
    c->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error);
    c->kernelsub = new_Kernel(cl->program,"sub",4,VOID_P,VOID_P,VOID_P,INT);
    return c;
}

Ponto3d __addLayer(Cnn c) {
    c->size += 1;
    c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
    Ponto3d in = c->sizeIn;
    if (c->size > 1) {
        in.x = c->camadas[c->size - 2]->saida->x;
        in.y = c->camadas[c->size - 2]->saida->y;
        in.z = c->camadas[c->size - 2]->saida->z;
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
        return c->err;

    }

    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createConv(c->cl,passo, tamanhoDoFiltro, numeroDeFiltros, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
                                         &c->parametros,&c->error,1);
    c->camadas[c->size - 1]->queue = c->queue;
    return 0;
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
    c->camadas[c->size - 1] = createPool(c->cl,passo, tamanhoDoFiltro, sizeIn.x, sizeIn.y, sizeIn.z, entrada,&c->parametros,&c->error);
    c->camadas[c->size - 1]->queue = c->queue;

    return 0;
}

int CnnAddReluLayer(Cnn c) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = creatRelu(c->cl,sizeIn.x, sizeIn.y, sizeIn.z, entrada,&c->error);
    c->camadas[c->size - 1]->queue = c->queue;
    return 0;
}

int CnnAddDropOutLayer(Cnn c, double pontoAtivacao,long long int seed) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createDropOut(c->cl,sizeIn.x, sizeIn.y, sizeIn.z, pontoAtivacao,seed,entrada,&c->error);
    c->camadas[c->size - 1]->queue = c->queue;
    return 0;
}

int CnnAddFullConnectLayer(Cnn c, UINT tamanhoDaSaida, Params *params, int funcaoDeAtivacao) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createFullConnect(c->cl,sizeIn.x, sizeIn.y, sizeIn.z, tamanhoDaSaida, entrada, params, funcaoDeAtivacao,1,&c->error);
    c->camadas[c->size - 1]->queue = c->queue;
    return 0;
}
int CnnCall(Cnn c,double *input){
    c->error.error = clEnqueueWriteBuffer(c->queue,c->camadas[0]->entrada->data,CL_TRUE,0,c->camadas[0]->entrada->bytes,input,0,NULL,NULL);
    for (int i = 0; i < c->size; ++i) {
        c->camadas[i]->ativa(c->camadas[i]);
    }
    return c->error.error;
}
int CnnLearn(Cnn c, double *target){
    if(c->size==0)return -1;
    Tensor lastGrad,targ;
    Tensor gradNext;
    lastGrad = newTensor(c->cl->context,c->camadas[c->size-1]->saida->x,c->camadas[c->size-1]->saida->y,c->camadas[c->size-1]->saida->z,&c->error);
    targ = newTensor(c->cl->context,c->camadas[c->size-1]->saida->x,c->camadas[c->size-1]->saida->y,c->camadas[c->size-1]->saida->z,&c->error);
    clEnqueueWriteBuffer(c->queue,targ->data,CL_TRUE,0,targ->bytes,target,0,NULL,NULL);

    int error = 0, id = 0;
    size_t global, local, resto;
    call_kernel(targ->x*targ->y*targ->z,
                Kernel_putArgs(&c->kernelsub, 4,&lastGrad->data,&c->camadas[c->size-1]->saida->data,&targ->data, &id);
                        error = clEnqueueNDRangeKernel(c->queue, c->kernelsub.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERRW(error, "falha ao chamar kernel sub")
    );
    gradNext = lastGrad;
    for (int l = c->size-1; l >=0; l--) {
        c->camadas[l]->calc_grads(c->camadas[l],gradNext);
        c->camadas[l]->corrige_pesos(c->camadas[l]);
        gradNext = c->camadas[l]->gradsEntrada;

    }
    releaseTensor(&lastGrad);
    releaseTensor(&targ);
}
void releaseCnn(Cnn *pc) {
    Cnn c = *pc;
    for (int i = 0; i < c->size; ++i) {
        c->camadas[i]->release(c->camadas + i);
    }
    free(c->camadas);
    clReleaseCommandQueue(c->queue);
    Kernel_release(&c->kernelsub);
    free(c);
    *pc = NULL;
}
void cnnSave(Cnn c,FILE *dst){
    for (int i = 0; i < c->size; ++i) {
        c->camadas[i]->salvar(c->cl,c->camadas[i],dst,&c->error);
    }
}
int cnnCarregar(Cnn c,FILE *src){
    if(c->size!= 0)return -1;
    Camada cm;
    Tensor entrada = NULL;
    while(1){
        cm = carregarCamada(c->cl,src,entrada,&c->parametros,&c->error);
        if(cm == NULL)return c->error.error;
        entrada = cm->saida;
        __addLayer(c);
        c->camadas[c->size-1] = cm;
    }
}
#endif