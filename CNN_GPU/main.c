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

int loadTargetData(double *target, size_t bytes, FILE *f, size_t *bytesReadd) {
    if (readBytes(f, (unsigned char *) target, bytes, bytesReadd) || !*bytesReadd)
        return 1;

    unsigned char *targint = (unsigned char *) target;
    for (int i = *bytesReadd - 1; i >= 0; i--) {
//        printf("%d: ", targint[i]);
        for (int j = 9; j >=0; j--) {
            target[i * 10 + j] = j == targint[i];
//            printf("%d %d %lf\n " ,j,targint[i], target[i * 10 + j] );

        }

    }
//    printf("\n");
    return 0;
}

int main() {
    srand(time(0));
    // criar  cnn
    Params p = {0.1, 0.99, 0.5};
    Cnn c = createCnnWithgpu("../kernels/gpu_function.cl", p, 28, 28, 1);
    CnnAddConvLayer(c, 1, 5, 8);
    CnnAddReluLayer(c);
    CnnAddPoolLayer(c, 2, 2);
    CnnAddFullConnectLayer(c, 10, FSIGMOIG);

    c->flags = CNN_FLAG_CALCULE_MAX | CNN_FLAG_CALCULE_ERROR;

    int tamanhoTensorEntrada = 28 * 28;
    int tamanhoTensorTarget = 10;
    int numeroDeImagens = 1000;
    size_t casos = 0;
    double *inputs = (double *) calloc(sizeof(double), tamanhoTensorEntrada * numeroDeImagens),
            *targets = (double *) calloc(sizeof(double), tamanhoTensorTarget * numeroDeImagens);
    int epoca = 0, maximoEpocas = 1;
    double erros = 0;

    FILE *f = fopen("../treino.txt", "w");
    FILE *imagens = fopen("../testes/train-images.idx3-ubyte", "rb");
    FILE *saidas = fopen("../testes/train-labels.idx1-ubyte", "rb");
    int i, r;
    fprintf(f, "Treino rede neural\n");
    char bufn[40] = {0};
    char b[16];
    for (; epoca < maximoEpocas; epoca++) {
        erros = 0;
        fprintf(f, "EPOCA %d\n", epoca);
        fread(b, 1, 16, imagens);
        fread(b, 1, 8, saidas);
        i = 0;
        while (!loadTargetData(targets, numeroDeImagens, saidas, &casos)) {
            normalizeImage(inputs, numeroDeImagens * tamanhoTensorEntrada, c->cl, c->queue, c->kerneldivInt, imagens, NULL);
//            printf("%d\n", casos);
            for (int caso = 0; caso < casos; caso++, i++) {
                CnnCall(c, inputs + tamanhoTensorEntrada * caso);
                CnnLearn(c, targets + tamanhoTensorTarget * caso);

                for (r = 0; r < 10; r++) {
                    if (*(targets + tamanhoTensorTarget * caso + r))break;
                }
//                snprintf(bufn, 40, "../testes/imgs/%d_%d_%d.pgm", i, r, (int) c->indiceSaida);
//                ppmp2(inputs + tamanhoTensorEntrada * caso, 28, 28, bufn);
//                fprintf(f, "%g\n\n ", c->indiceSaida);
                erros += c->normaErro;
            }
        }
        fprintf(f, "\nerror total %g, numero de casos %d\n\n", erros, i);
        fprintf(f, "------------------\n\n\n");
    }
    fclose(f);
    fclose(imagens);
    fclose(saidas);
    FILE *redeTreinada = fopen("../redeTreinada.cnn", "wb");
    cnnSave(c, redeTreinada);
    fclose(redeTreinada);

    if (inputs)free(inputs);
    if (targets)free(targets);
    releaseCnn(&c);
    return 0;
}
