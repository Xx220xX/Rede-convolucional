#include "libraryPythonWrapper.h"

void createCnnWrapper(Pointer *p, char *kernelFile, double hitLearn, double momento, double decaimentoDePeso, double multiplicador, UINT inx, UINT iny, UINT inz) {
    Params pr = {hitLearn, momento, decaimentoDePeso, multiplicador};
    Cnn c = createCnnWithgpu(kernelFile, pr, inx, iny, inz);
    p->p = c;

}

void releaseCnnWrapper(Pointer *p) {
    releaseCnn((Cnn *) &p->p);
}

#define atributeChecking(p, value)if(p!=0)*p = value

int CnnGetSize(Cnn c, int layer, int request, int *x, int *y, int *z, int *n) {
    if (layer < 0)layer = c->size + layer;
    if (layer < 0 || layer >= c->size)return -1;
    Tensor t;
    Camada cm = c->camadas[layer];
    switch (request) {
        case REQUEST_GRAD_INPUT:
        case REQUEST_INPUT:
            t = cm->entrada;
            break;
        case REQUEST_OUTPUT:
            t = cm->saida;
            break;
        case REQUEST_WEIGTH:
            switch (cm->type) {
                case RELU:
                case DROPOUT:
                case POOL:
                    return -2; // nao contem informacao
                case CONV:
                    atributeChecking(n, ((CamadaConv) cm)->numeroFiltros);
                    t = ((CamadaConv) cm)->filtros;
                    break;
                case FULLCONNECT:
                    t = ((CamadaFullConnect) cm)->pesos;
                    break;
                default:
                    return -5;//fatal error


            }
            break;
        default:
            return -3;//invalid request
    }
    atributeChecking(x, t->x);
    atributeChecking(y, t->y);
    atributeChecking(z, t->z);
    return 0;
}

int CnnGetTensorData(Cnn c, int layer, int request, int nfilter, double *dest) {
    if (layer < 0)layer = c->size + layer;
    if (layer < 0 || layer >= c->size)return -1;//invalid layer
    Tensor t;
    Camada cm = c->camadas[layer];
    size_t ofset = 0;
    switch (request) {
        case REQUEST_GRAD_INPUT:
            t = cm->gradsEntrada;
            break;
        case REQUEST_INPUT:
            t = cm->entrada;
            break;
        case REQUEST_OUTPUT:
            t = cm->saida;
            break;
        case REQUEST_WEIGTH:
            switch (cm->type) {
                case RELU:
                case DROPOUT:
                case POOL:
                    return -2; // nao contem informacao
                case CONV:
                    t = ((CamadaConv) cm)->filtros;
                    if (nfilter < 0 || nfilter >= ((CamadaConv) cm)->numeroFiltros)return -4;//invalid request2
                    ofset = t->bytes * nfilter;
                    break;
                case FULLCONNECT:
                    t = ((CamadaFullConnect) cm)->pesos;
                    break;
                default:
                    return -5;//fatal error
            }
            break;
        default:
            return -3;//invalid request
    }
    if (dest == NULL) return -5;//fatal error
    int error = clEnqueueReadBuffer(c->queue, t->data, CL_TRUE, ofset, t->bytes, dest, 0, NULL, NULL);
    clFinish(c->queue);
    if (error)return error - 100;
    return 0;
}

int CnnSaveInFile(Cnn c, char *fileName) {
    FILE *f = fopen(fileName, "wb");
    if (f == NULL)return -1;
    cnnSave(c, f);
    fclose(f);
    return c->error.error;
}

int CnnLoadByFile(Cnn c, char *fileName) {
    FILE *f = fopen(fileName, "rb");
    if (f == NULL)return -1;
    int err = cnnCarregar(c, f);
    fclose(f);
    return err;
}

int openFILE(Pointer *p, char *fileName, char *mode) {
    if (p == NULL)return -2;
    p->p = fopen(fileName, mode);
    if (p->p == NULL)return -1;
    return 0;
}

int closeFile(Pointer *p) {
    if (p == NULL)return -2;
    if (p->p == NULL)return -1;
    fclose(p->p);
    return 0;
}


void CnnInfo(Cnn c){
    printf("Numero de camadas %d\n",c->size);
    printf("Tamanho de entrada (%d,%d,%d)\n",c->sizeIn.x,c->sizeIn.y,c->sizeIn.z);
    printf("Camada\t saida\n");
    for(int i = 0;i<c->size;i++)
        printf("%d\t(%d,%d,%d)\n",i,c->camadas[i]->saida->x,c->camadas[i]->saida->y,c->camadas[i]->saida->z);

}

int getCnnError(Cnn c){
    return c->error.error;
}
void getCnnErrormsg(Cnn c,char *msg){
    memcpy(msg,c->error.msg,255);
}

void initRandom(long long int seed) {
    srand(seed);
}
