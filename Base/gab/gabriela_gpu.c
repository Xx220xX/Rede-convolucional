#include"gabriela_gpu.h"
#include <time.h>
#include "src/gabrielaLayer.h"


static WrapperCL wpcl;
static int gpu_init = 0;


void teste() {
    printf("c is okay\n");
}

void initGPU(const char *src) {
    if (gpu_init)return;
    WrapperCL_init(&wpcl, src);
    intern_set_seed(time(0));
    size_t maxLW = 1;
    int error = clGetDeviceInfo(wpcl.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLW, NULL);
    if (error)fprintf(stderr, "falha ao checar valor error id: %d\n", error);
    gab_set_max(maxLW);
    gpu_init = 1;

}

void initWithFile(const char *filename) {
    FILE *f;
    f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "arquivo nao encontrado no caminho %s\n", filename);
        return;
    }
    char *src = 0;
    long int size = 0;
    fseek(f, 0L, SEEK_END);
    size = ftell(f);
    src = calloc(size, sizeof(char));
    fseek(f, 0L, SEEK_SET);
    fread(src, sizeof(char), size, f);
    src[size - 1] = 0;
    initGPU(src);
    free(src);
}

void endGPU() {
    if (!gpu_init)return;
    gpu_init = 0;
    WrapperCL_release(&wpcl);
}

void call(Gab *p_gab, double *inp) {
    DNN_call(p_gab->gab, inp);
}

void learn(Gab *p_gab, double *trueOut) {
    DNN_learn(p_gab->gab, trueOut);
}

void release(Gab *p_gab) {
    DNN_release(p_gab->gab);
    free(p_gab->gab);
}

int create_DNN(Gab *p_gab, int *arq, int l_arq, int *funcs, char *norm, double hitLean) {
    p_gab->size = sizeof(DNN);
    int error = 0;
    DNN *dnn = calloc(1, p_gab->size);
    *dnn = new_DNN(&wpcl, arq, l_arq, funcs, norm, hitLean, &error);
    p_gab->gab = dnn;
    if (error) {
        fprintf(stderr, "falha ao criar rede neural\n");
        free(dnn);
        return error;
    }
    return 0;
}

void getoutput(Gab *p_gab, double *out) {
    DNN *gab = (DNN *) p_gab->gab;
    DNN_getA(gab, gab->L, out);
}

int getA(Gab *p_gab, int l, double *det) {
    return DNN_getA(p_gab->gab, l, det);
}

void setSeed(unsigned long int seed) {
    intern_set_seed(seed);
}

void checkLW() {
    size_t maxLW = 1;
    int error = clGetDeviceInfo(wpcl.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLW, NULL);
    if (error)fprintf(stderr, "falha ao checar valor error id: %d\n", error);
    else printf("max work group size: %zu\n", maxLW);

}

int randomize(Gab *p_gab) {
    int error = 0;
    DNN_randomize(p_gab->gab, &error);
    return error;
}

int sethitlearn(Gab *p_gab, double hl) {
    DNN_setHitlearn(p_gab->gab, hl);
    return 0;
}

void plot(DNN *self, char *file, char *mode) {
    FILE *f = fopen(file, mode);
    double *v = 0;
    for (int i = 0; i <= self->L; i++) {
        v = realloc(v, self->layers[i].a.bytes);
        DNN_getA(self, i, v);
        fprintf(f, "a%d = [", i);
        for (int j = 0; j < self->layers[i].a.m; j++) {
            fprintf(f, "%lf ", v[j]);
        }
        fprintf(f, "];\n");
        fprintf(f, "subplol(%d,1,%d)", self->L + 1, i + 1);
        fprintf(f, "stem(1:length(a%d),a%d);\n", i, i);
    }
    fclose(f);
    free(v);
}

#define op(a, b) ((int )a) ^ ((int )b)

void generateFigure(int A, int B,int w, Gab gab, FILE *f);

void testXor(char *file) {
    initWithFile(file);
    Gab gab;
//    setSeed(1);
    int arquitetura[] = {2, 1000, 200, 100, 20, 1};
    int funct[] = {TANH, TANH, TANH, TANH, TANH};
    char norm[] = {1, 0, 0, 0, 0};
//    char norm[] = {0,0,0,0};

    int la = sizeof(arquitetura) / sizeof(int);
    create_DNN(&gab, arquitetura, la, funct, norm, 0.03);
    double input[][2] = {{1, 1},
                         {0, 1},
                         {1, 0},
                         {0, 0}};
    double out[4][1];
    for (int i = 0; i < 4; ++i) out[i][0] = op(input[i][0], input[i][1]);
    int epocas = 0;
    int maxEpocas = 1000;
    double o = 0.0;
    double minEnergia = 1e-3;
    double energia = minEnergia + 1;
    clock_t t0 = clock();

    FILE *f = fopen("learn2.m", "w");
    fprintf(f, "clc;clear all;%% close all;\n");
    fprintf(f, "epc = 1:%d;\nenergia = [", maxEpocas);

    for (epocas = 0; epocas < maxEpocas; epocas++) {
        energia = 0;
        for (int i = 0; i < 4; ++i) {
            call(&gab, input[i]);
            learn(&gab, out[i]);
            getoutput(&gab, &o);
//            printf("%g xor %g =  %.4g\n",input[i][0],input[i][1],o);
            energia += (o - out[i][0]) * (o - out[i][0]);
        }
        energia /= 2.0;
        fprintf(f, "%lf ", energia);
//        printf("Energia %lf\n",energia);
    }
    fprintf(f, "];\n");
    fprintf(f, "plot(epc,energia)\n");

    t0 = clock() - t0;
    printf("depois de %d epocas a energia foi %lf\n", epocas, energia);
    printf("tempo total gasto %ld ms\n", t0);
    printf("tempo medio gasto por exemplo %lf ms\n", t0 / ((double) epocas * 4.0));


    /// fazendo grafico


    fclose(f);
    release(&gab);
    endGPU();
}

void generateFigure(int A, int B,int w, Gab gab, FILE *f) {
    double dx = ((A - B)>0?(A-B):-(A-B)) / (double) w;
    double in[2];
    double o;

    fprintf(f, "region = [");
    for (int i = 0; i < w; i++) {
        in[0] = i * dx + A;
        for (int j = 0; j < w; j++) {
            in[1] = (w - j) * dx + A;
            call(&gab, in);
            getoutput(&gab, &o);
            fprintf(f, "%lf ", o);

        }
        if (i + 1 < w)
            fprintf(f, "\n");
    }
    fprintf(f, "];\n");
    fprintf(f, "im = zeros(%d,%d,3);\n", w, w);
    fprintf(f, "im(:,:,1) = (region>.5)*255;im(:,:,3) = (region<.5)*255;\n");
    fprintf(f, "im = uint8(im);\n");

    fprintf(f, "figure;plot(epc,energia)\n");
    fprintf(f, "figure;imshow(im)\n");

}