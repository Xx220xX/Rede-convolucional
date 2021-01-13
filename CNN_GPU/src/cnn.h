#ifndef CNN_GPU_CNN_H
#define CNN_GPU_CNN_H

#include "Camada.h"
#include "CamadaConv.h"
#include "CamadaRelu.h"
#include "CamadaDropOut.h"
#include "CamadaFullConnect.h"
#include "CamadaPool.h"


#define INVALID_FILTER_SIZE -1

typedef struct _cnn{
    Params parametros;
    Camada *camadas;
    int size;
    Ponto3d sizeIn;
    char error;
}  TypeCnn;
typedef TypeCnn  * Cnn;
Cnn createCnn(Params p, UINT inx, UINT iny, UINT inz) {
    Cnn c = (Cnn) calloc(1, sizeof(TypeCnn));
    c->parametros = p;
    c->sizeIn = (Ponto3d) {inx, iny, inz};
    return c;
}

Ponto3d __addLayer(Cnn c) {
    c->size += 1;
    c->camadas = (Camada *) realloc(c->camadas, c->size * sizeof(Camada));
    Ponto3d in = c->sizeIn;
    if (c->size > 1) {
        in.x = c->camadas[c->size - 2]->saida->tx;
        in.y = c->camadas[c->size - 2]->saida->ty;
        in.z = c->camadas[c->size - 2]->saida->tz;
    }
    return in;
}

#define checkSizeFilter(v,tam,pas) (((v)-(tam))/(pas)) ==((double)(v)-(tam))/((double)(pas))
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
    if(!checkSizeFilter(sizeIn.x,tamanhoDoFiltro,passo) || !checkSizeFilter(sizeIn.y,tamanhoDoFiltro,passo) ){
        c->error = INVALID_FILTER_SIZE;
        c->size --;
        c->camadas = (Camada*) realloc(c->camadas, c->size * sizeof(Camada));
        return c->error;

    }

    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createConv(passo, tamanhoDoFiltro, numeroDeFiltros, sizeIn.x, sizeIn.y, sizeIn.z, entrada,
                                         &c->parametros);
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
    if(!checkSizeFilter(sizeIn.x,tamanhoDoFiltro,passo) || !checkSizeFilter(sizeIn.y,tamanhoDoFiltro,passo) ){
        c->error = INVALID_FILTER_SIZE;
        c->size --;
        c->camadas = (Camada*) realloc(c->camadas, c->size * sizeof(Camada));
        return c->error;

    }

    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createPool(passo, tamanhoDoFiltro,  sizeIn.x, sizeIn.y, sizeIn.z, entrada,
                                         &c->parametros);
    return 0;
}

int CnnAddReluLayer(Cnn c) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = creatRelu(sizeIn.x, sizeIn.y, sizeIn.z, entrada);
    return 0;
}

int CnnAddDropOutLayer(Cnn c, double pontoAtivacao) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createDropOut(sizeIn.x, sizeIn.y, sizeIn.z, pontoAtivacao, entrada);
    return 0;
}

int CnnAddFullConnectLayer(Cnn c, UINT tamanhoDaSaida,int funcaoDeAtivacao) {
    Ponto3d sizeIn = __addLayer(c);
    Tensor entrada = NULL;
    if (c->size > 1)entrada = c->camadas[c->size - 2]->saida;
    c->camadas[c->size - 1] = createFullConnect(sizeIn.x, sizeIn.y, sizeIn.z, tamanhoDaSaida, entrada,funcaoDeAtivacao);
    return 0;
}

void releaseCnn(Cnn *pc) {
    Cnn c = *pc;
    for (int i = 0; i < c->size; ++i) {
        c->camadas[i]->release(c->camadas + i);
    }
    free(c->camadas);
    free(c);
    *pc = NULL;
}

#endif