//#define LOG_CNN_ADD_LAYERS

#include "src/cnn.h"

void ppmp2(double *data, int x, int y, char *fileName) {
    FILE *f = fopen(fileName, "w");
    fprintf(f, "P2\n");
    fprintf(f, "%d %d\n", y, x);
    fprintf(f, "255\n");
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; ++j) {
            fprintf(f, "%d", (int) (data[i * y + j] * 255));
            if (j < y - 1)fprintf(f, " ");
        }
        if (i < x - 1)
            fprintf(f, "\n");
    }
    fclose(f);
}

int readBytes(FILE *f, unsigned char *buff, size_t bytes, size_t *bytesReaded) {
    if (feof(f)) {
        fseek(f, 0, SEEK_SET);
        return 1;
    }
    size_t a = 0;
    if (!bytesReaded) {
        bytesReaded = &a;
    }
    *bytesReaded = fread(buff, 1, bytes, f);
    return 0;
}

int normalizeImage(double *imagem, size_t bytes, WrapperCL *cl, cl_command_queue queue, Kernel divInt, FILE *f, size_t *bytesReadd) {

    if (readBytes(f, (unsigned char *) imagem, bytes, bytesReadd)) {
        return 1;
    }
    cl_mem mInt, mDou;
    int error = 0;
    mInt = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes, NULL, &error);
    mDou = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes * sizeof(double), NULL, &error);
    dividirVetorInt((unsigned char *) imagem, imagem, mInt, mDou, bytes, 255, divInt, queue);
    clReleaseMemObject(mInt);
    clReleaseMemObject(mDou);
    return 0;
}

int loadTargetData(double *target, size_t bytes, WrapperCL *cl, cl_command_queue queue, Kernel int2vector, FILE *f, size_t *bytesReadd) {
    if (readBytes(f, (unsigned char *) target, bytes, bytesReadd) || !*bytesReadd)
        return 1;
    cl_mem mInt, mDou;
    int error = 0;
    mInt = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes, NULL, &error);
    mDou = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes *10* sizeof(double), NULL, &error);
    int2doubleVector(cl,(unsigned char *) target, target, mInt, mDou, *bytesReadd, 10, int2vector, queue);
    clReleaseMemObject(mInt);
    clReleaseMemObject(mDou);
    return 0;
}

int main() {
    srand(time(0));
    // criar  cnn
    Params p = {0.1, 0.0, 0.5};
    Cnn c = createCnnWithgpu("../kernels/gpu_function.cl", p, 28, 28, 1);
    CnnAddConvLayer(c, 1, 5, 18);
    CnnAddReluLayer(c);
    CnnAddPoolLayer(c, 2, 2);
    CnnAddFullConnectLayer(c, 50, FSIGMOIG);
    CnnAddFullConnectLayer(c, 10, FSIGMOIG);

    c->flags = CNN_FLAG_CALCULE_MAX | CNN_FLAG_CALCULE_ERROR;


    int maximoEpocas = 5;
    int buffImageSize = 1000;
    int limiteImages = 1000;


    int tamanhoTensorEntrada = 28 * 28;
    int tamanhoTensorTarget = 10;
    size_t casos = 0;
    double *inputs = (double *) calloc(sizeof(double), tamanhoTensorEntrada * buffImageSize),
            *targets = (double *) calloc(sizeof(double), tamanhoTensorTarget * buffImageSize);
    int epoca = 0;
    double erros = 0;
    int acertos = 0;

    FILE *f = fopen("../treino.txt", "w");
    FILE *imagens = fopen("../testes/train-images.idx3-ubyte", "rb");
    FILE *saidas = fopen("../testes/train-labels.idx1-ubyte", "rb");
    int i, r,t;
    fprintf(f, "Treino rede neural, numero de threads disponiveis %d\n", c->cl->maxworks);
    fprintf(f, "numero de epocas %d, tamanho do buff %d\n", maximoEpocas, buffImageSize);
    char b[16];
    double out[10];
    size_t initTimeALL = time(0), initTimeLocal;
    for (; epoca < maximoEpocas; epoca++) {
        initTimeLocal = time(0);
        erros = 0;
        acertos =0;
        fprintf(f, "EPOCA %d\n", epoca);
        fclose(imagens);
        fclose(saidas);
        imagens = fopen("../testes/train-images.idx3-ubyte", "rb");
        saidas = fopen("../testes/train-labels.idx1-ubyte", "rb");
        fread(b, 1, 16, imagens);
        fread(b, 1, 8, saidas);
        i = 0;

        while (i < limiteImages && !loadTargetData(targets, buffImageSize, c->cl, c->queue, c->kernelInt2Vector, saidas, &casos)) {
            normalizeImage(inputs, buffImageSize * tamanhoTensorEntrada, c->cl, c->queue, c->kerneldivInt, imagens, NULL);
            for (int caso = 0; caso < casos && i < limiteImages; caso++, i++) {
                CnnCall(c, inputs + tamanhoTensorEntrada * caso);
                CnnLearn(c, targets + tamanhoTensorTarget * caso);
                Cnngetout(c, out);
                for (t = 0; t < 9 &&! *(targets + tamanhoTensorTarget * caso + t) ; t++);
                for (r = 0; r < 9 &&! *(out +r) ; r++);
                if(r==t){
                    acertos++;
                }
                erros += c->normaErro;
            }
            printf("foram lidas %d imagens\n", casos);
        }

        fprintf(f, "\nerror euclidiano  total %g, numero de casos %d  numero de acertos %d tempo gasto %llu seg\n\n", erros, i,acertos,time(0) - initTimeLocal);
        fprintf(f, "------------------\n\n\n");
        fclose(f);
        f = fopen("../treino.txt", "a");

    }
    fprintf(f, "Tempo gasto para %d epocas %lf min\n", maximoEpocas, (time(0) - initTimeALL) / 60.0);
    fclose(f);
    fclose(imagens);
    fclose(saidas);
    FILE *redeTreinada = fopen("../../webCreatorImg/redeTreinada.cnn", "wb");
    cnnSave(c, redeTreinada);
    fclose(redeTreinada);
    if (inputs)free(inputs);
    if (targets)free(targets);
    releaseCnn(&c);
    return 0;
}
