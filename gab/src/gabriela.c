//
// Created by Xx220xX on 12/05/2020.
//

#include "gabriela.h"


static LCG lcg = {
        0x5DEECE66DULL,
        11ULL,
        1ULL << 48,
        (1ULL << 48) - 1,
        1 << (sizeof(int) * 8 - 1),
        0
};

static int max_works = 1024;

static int max_one_works = 32;


DNN new_DNN(WrapperCL *wrp, int *n, int ln, double hit_learn, int *err) {
    int error = 0;
    if (!err)err = &error;
    DNN self = {0};
    self.API_CL = *wrp;
    self.queue = clCreateCommandQueueWithProperties(self.API_CL.context, self.API_CL.device, NULL, &error);
    *err = error;
    PER(error, "Nao foi possivel criar queue")
    self.n = (cl_int *) calloc(ln, sizeof(cl_int));
    memcpy(self.n, n, ln * sizeof(int));

    self.L = ln - 1;
    self.hLearn = hit_learn;


    self.w = (Mat *) (calloc(self.L + 1, sizeof(Mat)));
    self.b = (Mat *) (calloc(self.L + 1, sizeof(Mat)));
    self.z = (Mat *) (calloc(self.L + 1, sizeof(Mat)));
    self.a = (Mat *) (calloc(self.L + 1, sizeof(Mat)));
    self.dw = (Mat *) (calloc(self.L + 1, sizeof(Mat)));
    self.dz = (Mat *) (calloc(self.L + 1, sizeof(Mat)));

    self.y = new_Mat(self.API_CL.context, n[self.L], 1, &error);

    self.a[0] = new_Mat(self.API_CL.context, n[0], 1, &error);

    for (int i = 1; i < ln && !error; i++) {
        self.w[i] = new_Mat(self.API_CL.context, n[i], n[i - 1], &error);

        self.b[i] = new_Mat(self.API_CL.context, n[i], 1, &error);
        PER(error, "falha ao criar matriz b")

        self.z[i] = new_Mat(self.API_CL.context, n[i], 1, &error);
        PER(error, "falha ao criar matriz z")

        self.a[i] = new_Mat(self.API_CL.context, n[i], 1, &error);
        PER(error, "falha ao criar matriz a")


        self.dw[i] = new_Mat(self.API_CL.context, n[i], n[i - 1], &error);
        PER(error, "falha ao criar matriz dw")

        self.dz[i] = new_Mat(self.API_CL.context, n[i], 1, &error);
        PER(error, "falha ao criar matriz dz")

    }
    *err = error;
    PER(error, "falha ao criar matrizes")
    self.out = (double *) calloc(n[self.L], sizeof(double));
    DNN_randomize(&self, &error);
    *err = error;
    PER(error, "falha ao randomizar matrizes")

// w,a-,ma,b,z,a,i0
    self.kernelCall = new_Kernel(self.API_CL.program, "gab_call", 7, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/,
                                 VOID_P/*z*/,
                                 VOID_P/*a*/,
                                 INT/*i0*/);
/*dz[L] = a[L] - y
 * b[L] = b[L] - hit*dz[L}*/
/**
 * void gab_sub(double *aL, double *y, double *dzL, double *bL, double hit)
 */
    self.kernelLearn_aL_minus_y = new_Kernel(self.API_CL.program, "gab_sub", 6,
                                             VOID_P/*a[l]*/, VOID_P/*y[L]*/,
                                             VOID_P/*dz[L]*/,
                                             VOID_P/*b[L]*/,
                                             DOUBLE /*hit*/, INT/*i0*/);

/*
 * dw[l] = dz[l]*a[l-1]^t
 */
    self.kernelLearn_dzl_ast_al_down = new_Kernel(self.API_CL.program, "gab_mult_w_a", 5, VOID_P/*dz[l]*/,
                                                  VOID_P/*a[l-1]*/,
                                                  INT/*a[l-1].m*/,
                                                  VOID_P/*w[l]*/, INT/*i0*/);


    self.kernelLearn_wupT_ast_dzup = new_Kernel(self.API_CL.program, "gab_hidem_dz", 8, VOID_P/*w[l+1]*/,
                                                VOID_P/*dz[l+1]*/, INT/*mdz[l+1]*/, VOID_P/*z[l]*/, VOID_P/*dz[l]*/,
                                                VOID_P/*b[l]*/, DOUBLE/*hit*/, INT/*i0*/ );

    self.kernelUpdateWeight = new_Kernel(self.API_CL.program, "gab_update_w", 4, VOID_P/*w[l]*/, VOID_P/*dw[l]*/,
                                         DOUBLE/*hitLearn*/, INT/*i0*/);

    *err = error;
    if (error) {
        fprintf(stderr, "error enquanto criava RNN %d\n", error);
        free(self.n);
        free(self.out);
        free(self.w);
        free(self.b);
        free(self.z);
        free(self.a);
        free(self.dw);
        free(self.dz);
    }
    return self;
}


#define call_kernel(total, command)\
    id = 0;\
    global =local= (total);\
    if(global<max_works){\
        command\
    }else{\
        resto = global % max_works;\
        global = (global / max_works) * max_works;\
        local = max_works;\
        command\
        if(resto){\
             id = global;\
             global = local = resto;\
             command\
        }\
    }

int DNN_call(DNN *self, double *input) {
    int error = 0;
    error = clEnqueueWriteBuffer(self->queue, self->a[0].v, CL_TRUE, 0, self->a[0].bytes, input, 0, NULL, NULL);
    PERR(error, "falha ao escrever no buffer a[0]")
    // z_1 = W*A + B
    // a_1 = f(z)
    //-0.542982
    size_t global, local = 1, resto;
    int id = 0;
    for (int l = 1; l <= self->L; ++l) {
        call_kernel(self->w[l].m,
                    Kernel_putArgs(&self->kernelCall, 7, &self->w[l].v, &self->a[l - 1].v, &self->a[l - 1].m, &self->b[l].v,
                                   &self->z[l].v, &self->a[l].v, &id);
                            error = clEnqueueNDRangeKernel(self->queue, self->kernelCall.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                            PERR(error, "erro ao chamar kernel call")
        )

    }

    error = clFinish(self->queue);
    PERR(error, "error on clfinish ")

    error = clEnqueueReadBuffer(self->queue, self->a[self->L].v, CL_TRUE, 0, self->a[self->L].bytes, self->out, 0, NULL,
                                NULL);
    PERR(error, "error on read buffer ")
}


int DNN_learn(DNN *self, double *trueout) {
    int error = 0, id = 0;
    size_t global, local, resto;
    error = clEnqueueWriteBuffer(self->queue, self->y.v, CL_TRUE, 0, self->y.bytes, trueout, 0, NULL, NULL);
    PERR(error, "falha ao escrever no buffer y")

    // WORKS(self->y.m, 0, self->y.lw_m, 0);
    call_kernel(self->y.m,
                Kernel_putArgs(&self->kernelLearn_aL_minus_y, 6, &self->a[self->L].v, &self->y.v, &self->dz[self->L].v,
                               &self->b[self->L].v, &self->hLearn, &id);
                        error = clEnqueueNDRangeKernel(self->queue, self->kernelLearn_aL_minus_y.kernel, 1, NULL, &global, &local, 0, NULL,
                                                       NULL);
                        PERR(error, "falha ao chamar kernel AL-Y")
    )

//    WORKS(self->dz[self->L].m, self->a[self->L - 1].m, self->dz[self->L].lw_m, self->a[self->L - 1].lw_m);
    call_kernel(self->dz[self->L].m * self->a[self->L - 1].m,
                Kernel_putArgs(&self->kernelLearn_dzl_ast_al_down, 5, &self->dz[self->L].v, &self->a[self->L - 1].v, &self->a[self->L - 1].m, &self->dw[self->L].v, &id);
                        error = clEnqueueNDRangeKernel(self->queue, self->kernelLearn_dzl_ast_al_down.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                        PERR(error, "falha ao chamar kernel learn last")
    )

    for (int l = self->L - 1; l > 0; l--) {
        call_kernel(self->w[l + 1].n,

                    Kernel_putArgs(&self->kernelLearn_wupT_ast_dzup, 8, &self->w[l + 1].v, &self->dz[l + 1].v, &self->dz[l + 1].m, &self->z[l].v, &self->dz[l].v, &self->b[l].v, &self->hLearn, &id);
                            error = clEnqueueNDRangeKernel(self->queue, self->kernelLearn_wupT_ast_dzup.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                            PERR(error, "Falha ao chamar kernel wuT*dzUp")
        )

//        WORKS(self->dz[l].m, self->a[l - 1].m, self->dz[l].lw_m, self->a[l - 1].lw_m);
        call_kernel(self->dz[l].m * self->a[l - 1].m,
                    Kernel_putArgs(&self->kernelLearn_dzl_ast_al_down, 5, &self->dz[l].v, &self->a[l - 1].v, &self->a[l - 1].m, &self->dw[l].v, &id);
                            error = clEnqueueNDRangeKernel(self->queue, self->kernelLearn_dzl_ast_al_down.kernel, 1, NULL, &global, &local, 0,
                                                           NULL,
                                                           NULL);
                            PERR(error, "falha ao chamar kernel dzl * aL")
        )

    }
    for (int l = 1; l <= self->L; ++l) {
//        WORKS(self->w[l].m * self->w[l].n, 0, self->w[l].lw_m_only, 0);/
        call_kernel(self->w[l].m * self->w[l].n,
                    Kernel_putArgs(&self->kernelUpdateWeight, 4, &self->w[l].v, &self->dw[l].v, &self->hLearn, &id);

                            error = clEnqueueNDRangeKernel(self->queue, self->kernelUpdateWeight.kernel, 1, NULL, &global, &local, 0, NULL,
                                                           NULL);
                            PERR(error, "falha ao chamar kernel update")
        )
    }
    error = clFinish(self->queue);
    PERR(error, "erro clfinish")
    return 0;

}

void DNN_randomize(DNN *self, int *err) {
    double *p;
    double maximo = 2,       minimo = -1;
    if (err && *err)return;
    int error = 0;
    if (!err)err = &error;
    for (int i = 1; i <= self->L; i++) {
        p = (double *) calloc(self->w[i].bytes, 1);
        for (int j = 0; j < self->w[i].m * self->w[i].n; ++j) p[j] = LCG_randD(&lcg) * maximo + minimo;
        *err = clEnqueueWriteBuffer(self->queue, self->w[i].v, CL_TRUE, 0, self->w[i].bytes, p, 0, NULL, NULL);
        if (self->w[i].bytes < self->b[i].bytes) {
            free(p);
            p = (double *) calloc(self->b[i].bytes, 1);
        }
        for (int j = 0; j < self->b[i].m; ++j) p[j] = LCG_randD(&lcg) * maximo + minimo;
        *err = clEnqueueWriteBuffer(self->queue, self->b[i].v, CL_TRUE, 0, self->b[i].bytes, p, 0, NULL, NULL);
        free(p);
    }

}

void DNN_release(DNN *self) {
    Mat_release(&self->y);
    Mat_release(&self->a[0]);
    for (int i = 1; i <= self->L; i++) {
        Mat_release(&self->w[i]);
        Mat_release(&self->b[i]);
        Mat_release(&self->z[i]);
        Mat_release(&self->a[i]);
        Mat_release(&self->dw[i]);
        Mat_release(&self->dz[i]);
    }

    free(self->w);
    free(self->b);
    free(self->z);
    free(self->a);
    free(self->dw);
    free(self->dz);

    self->w = self->b = self->z = self->a = self->dw = self->dz = NULL;

    free(self->n);
    free(self->out);
    self->n = NULL;
    self->out = NULL;

    Kernel_release(&self->kernelCall);
    Kernel_release(&self->kernelLearn_aL_minus_y);
    Kernel_release(&self->kernelLearn_dzl_ast_al_down);
    Kernel_release(&self->kernelLearn_wupT_ast_dzup);
    Kernel_release(&self->kernelUpdateWeight);

    clReleaseCommandQueue(self->queue);
    self->queue = NULL;
}

void intern_setSeed(Bytes8 seed) {
    LCG_setSeed(&lcg, seed);
}

void gab_set_max(int max) {
    max_works = max;
    max_one_works = sqrt(max);
    printf("maximo works group set to: %d %d\n", max, max_one_works);
}

void DNN_setHitlearn(DNN *self, double hl) {
    self->hLearn = hl;
}

int DNN_getA(DNN *self, int l, double *det) {
    if (l < 0 || l > self->L)return -1;
    int error = clEnqueueReadBuffer(self->queue, self->a[l].v, CL_TRUE, 0, self->a[l].bytes, det, 0, NULL,
                                    NULL);
    return error;
}

