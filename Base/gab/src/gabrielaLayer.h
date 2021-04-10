//
// Created by Xx220xX on 28/07/2020.
//

#ifndef GAB_GABRIELALAYER_H
#define GAB_GABRIELALAYER_H

#include<CL/cl.h>
#include "WrapperCL.h"
#include "matGPU.h"
#include "lcg.h"
#include <math.h>
#define NUMERO_TOTAL_DE_KERNEL     31
// ###### KERNELS KEY CODE ####
#define KERNEL_FULL_FEED            0
#define KERNEL_WA_B_FEED            6
#define KERNEL_NORMALIZE            7

#define KERNEL_SUM                 13
#define KERNEL_STD                 14
#define KERNEL_DIVIDE_VECTOR       15

#define KERNEL_FIND_LAST_DZL       16
#define KERNEL_FIND_DZL            17
#define KERNEL_FIND_DZL_NORMALIZE  23
#define KERNEL_FIND_DWL            29
#define KERNEL_FIND_AND_UPDATE_DWL 30



// ###### functions ID    ####
#define ALAN     0
#define TANH     1
#define RELU     2
#define SIGMOID  3
#define SOFTMAX  4
#define IDENTIFY 5


typedef int (*f_forward)(void *, void *, void *);
typedef int (*f_backward)(void *, void *, void *,void *);

typedef struct {
    Mat w, a, b, z;
    Mat dz, dw;
    int normalize, funcao_de_ativacao;
    cl_double media, desvio_padrao;
    f_forward forward;
    f_backward backward;
} Layer;

typedef struct {
    cl_int *n;
    cl_int L;
    cl_double hLearn;
    cl_command_queue queue;
    Kernel *all_kernels;
    Layer *layers;
    Mat y;
    WrapperCL API_CL;
    double *out;
    int *functions;
    char *normalize;
} DNN;

DNN new_DNN(WrapperCL *wrp, int *n, int ln, int *functions, char *normalize, double hit_learn, int *err);

void DNN_release(DNN *self);

int DNN_call(DNN *self, double *input);

int DNN_learn(DNN *self, double *trueout);

void DNN_randomize(DNN *self, int *err);

void intern_set_seed(Bytes8 seed);

void DNN_setHitlearn(DNN *self, double hl);

void gab_set_max(int );
int DNN_getA(DNN *self, int l, double *det);

int saveDNN(DNN*, char *);
DNN loadDNN(WrapperCL *wpr,char *);

#endif //GAB_GABRIELALAYER_H
