//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_GABRIELA_H
#define GAB_GABRIELA_H

#include<CL/cl.h>
#include "WrapperCL.h"
#include "matGPU.h"
#include "lcg.h"
#include <math.h>

typedef struct {
    cl_int *n;
    cl_int L;
    cl_double hLearn;
    Kernel  kernelCall,
            kernelLearn_aL_minus_y,
            kernelLearn_dzl_ast_al_down,
            kernelLearn_wupT_ast_dzup,
            kernelUpdateWeight;
    cl_command_queue queue;

    Mat *w, *a, *b, *z;
    Mat *dw, *dz;
    Mat y;
    WrapperCL API_CL;
    double *out;

} DNN;

DNN new_DNN(WrapperCL *wrp, int *n, int ln, double hit_learn, int *err);

int DNN_call(DNN *self, double *input);

int DNN_learn(DNN *self, double *trueout);

void DNN_randomize(DNN *self, int *err);

void DNN_release(DNN *self);

void intern_setSeed(Bytes8 seed);

void DNN_setHitlearn(DNN *self,double hl);

void gab_set_max(int max);

int DNN_getA(DNN*self, int l,double *det);
#endif //GAB_GABRIELA_H
