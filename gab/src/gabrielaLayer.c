//
// Created by Xx220xX on 28/07/2020.
//

#include "gabrielaLayer.h"

static LCG lcg = {0x5DEECE66DULL,
                  11ULL,
                  1ULL << 48,
                  (1ULL << 48) - 1,
                  1,
                  1 << (sizeof(int) * 8 - 1)

};
static int max_works = 1024;

//## functions used for forward and back

int forward(void *pdnn, void *pinp, void *player);

int forward_normalize(void *pdnn, void *pinp, void *player);

int backward_first_layer(void *, void *, void *, void *);

int backward(void *, void *, void *, void *);

int backward_last_layer(void *, void *, void *, void *);

int backward_normalize(void *, void *, void *, void *);

int backward_first_layer_normalize(void *pdnn, void *pl0, void *pl, void *pl1);

double kernel_sum(DNN *, Mat);
//## functions used for forward and back


Layer new_Layer(cl_context context, int inp, int out, int func_id, int normalize, int islastorfirst, int *erro);

void Layer_randomize(DNN *self, Layer *l);

double rando(double max, double min);

double kernel_media(DNN *, Mat);

double kernel_desvio_padrao(DNN *self, Mat x, double media);

DNN new_DNN(WrapperCL *wrp, int *n, int ln, int *functions, char *normalize, double hit_learn, int *err) {
    int error = 0;
    if (!err)err = &error;
    DNN self = {0};
    self.API_CL = *wrp;
    self.queue = clCreateCommandQueueWithProperties(self.API_CL.context, self.API_CL.device, NULL, &error);
    *err = error;
    PER(error, "Nao foi possivel criar queue")
    self.n = (cl_int *) calloc(ln, sizeof(cl_int));
    self.functions = (cl_int *) calloc(ln, sizeof(cl_int));
    self.normalize = (char *) calloc(ln, sizeof(char));
    memcpy(self.n, n, ln * sizeof(int));
    if (functions)
        memcpy(self.functions + 1, functions, (ln - 1) * sizeof(int));
    if (normalize)
        memcpy(self.normalize + 1, normalize, (ln - 1) * sizeof(char));

    self.L = ln - 1;
    self.hLearn = hit_learn;
    self.layers = (Layer *) (calloc(self.L + 1, sizeof(Layer)));

    self.y = new_Mat(self.API_CL.context, n[self.L], 1, &error);
    self.layers[0] = new_Layer(self.API_CL.context, 0, n[0], 0, 0, 0, &error);
    for (int i = 1; i < ln && !error; i++) {
        self.layers[i] = new_Layer(self.API_CL.context, n[i - 1], n[i], self.functions[i], self.normalize[i], i == 1 ? -1 : ((i == ln - 1) ? 1 : 0), &error);
        PER(error, "falha ao layer")
    }
    *err = error;
    PER(error, "falha ao criar matrizes")
    self.out = (double *) calloc(n[self.L], sizeof(double));
    DNN_randomize(&self, &error);
    *err = error;
    PER(error, "falha ao randomizar matrizes")

    self.all_kernels = (Kernel *) calloc(NUMERO_TOTAL_DE_KERNEL, sizeof(Kernel));

    self.all_kernels[KERNEL_FULL_FEED + ALAN] = new_Kernel(self.API_CL.program, "gab_feed_alan", 7, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/, VOID_P/*z*/, VOID_P/*a*/, INT/*i0*/);
    self.all_kernels[KERNEL_FULL_FEED + TANH] = new_Kernel(self.API_CL.program, "gab_feed_tanh", 7, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/, VOID_P/*z*/, VOID_P/*a*/, INT/*i0*/);
    self.all_kernels[KERNEL_FULL_FEED + RELU] = new_Kernel(self.API_CL.program, "gab_feed_relu", 7, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/, VOID_P/*z*/, VOID_P/*a*/, INT/*i0*/);
    self.all_kernels[KERNEL_FULL_FEED + SIGMOID] = new_Kernel(self.API_CL.program, "gab_feed_sigmoid", 7, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/, VOID_P/*z*/, VOID_P/*a*/, INT/*i0*/);
    self.all_kernels[KERNEL_FULL_FEED + IDENTIFY] = new_Kernel(self.API_CL.program, "gab_feed_identify", 7, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/, VOID_P/*z*/, VOID_P/*a*/, INT/*i0*/);
    self.all_kernels[KERNEL_FULL_FEED + SOFTMAX] = new_Kernel(self.API_CL.program, "gab_feed_softmax", 7, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/, VOID_P/*z*/, VOID_P/*a*/, INT/*i0*/);

    self.all_kernels[KERNEL_WA_B_FEED] = new_Kernel(self.API_CL.program, "gab_feed", 6, VOID_P/*w*/, VOID_P/*a-*/, INT/*ma-*/, VOID_P/*b*/, VOID_P/*z*/, INT/*i0*/);

    self.all_kernels[KERNEL_NORMALIZE + ALAN] = new_Kernel(self.API_CL.program, "gab_normalize_alan", 5, VOID_P/*a*/, VOID_P/*z*/, DOUBLE/*media*/, DOUBLE/*desvio_padrao*/, INT/*i0*/);
    self.all_kernels[KERNEL_NORMALIZE + TANH] = new_Kernel(self.API_CL.program, "gab_normalize_tanh", 5, VOID_P/*a*/, VOID_P/*z*/, DOUBLE/*media*/, DOUBLE/*desvio_padrao*/, INT/*i0*/);
    self.all_kernels[KERNEL_NORMALIZE + RELU] = new_Kernel(self.API_CL.program, "gab_normalize_relu", 5, VOID_P/*a*/, VOID_P/*z*/, DOUBLE/*media*/, DOUBLE/*desvio_padrao*/, INT/*i0*/);
    self.all_kernels[KERNEL_NORMALIZE + SIGMOID] = new_Kernel(self.API_CL.program, "gab_normalize_sigmoid", 5, VOID_P/*a*/, VOID_P/*z*/, DOUBLE/*media*/, DOUBLE/*desvio_padrao*/, INT/*i0*/);
    self.all_kernels[KERNEL_NORMALIZE + IDENTIFY] = new_Kernel(self.API_CL.program, "gab_normalize_identify", 5, VOID_P/*a*/, VOID_P/*z*/, DOUBLE/*media*/, DOUBLE/*desvio_padrao*/, INT/*i0*/);
    self.all_kernels[KERNEL_NORMALIZE + SOFTMAX] = new_Kernel(self.API_CL.program, "gab_normalize_softmax", 5, VOID_P/*a*/, VOID_P/*z*/, DOUBLE/*media*/, DOUBLE/*desvio_padrao*/, INT/*i0*/);


    self.all_kernels[KERNEL_SUM] = new_Kernel(self.API_CL.program, "gab_sum", 3, VOID_P/*ans*/, VOID_P/*x*/, INT/*n*/);
    self.all_kernels[KERNEL_STD] = new_Kernel(self.API_CL.program, "gab_desvio_padrao", 4, VOID_P/*ans*/, VOID_P/*x*/, DOUBLE /*media*/, INT/*n*/);
    self.all_kernels[KERNEL_DIVIDE_VECTOR] = new_Kernel(self.API_CL.program, "gab_divide_vector", 4, VOID_P/*ans*/, VOID_P/*x*/, DOUBLE /*value*/, INT/*i0*/);


    self.all_kernels[KERNEL_FIND_LAST_DZL] = new_Kernel(self.API_CL.program, "gab_find_last_dzl", 6, VOID_P/*aL*/, VOID_P/*y*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE /*hitlearn*/, INT/*i0*/);
    self.all_kernels[KERNEL_FIND_DZL + ALAN] = new_Kernel(self.API_CL.program, "gab_find_dzl_alan", 10, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL + TANH] = new_Kernel(self.API_CL.program, "gab_find_dzl_tanh", 10, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL + RELU] = new_Kernel(self.API_CL.program, "gab_find_dzl_relu", 10, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL + SIGMOID] = new_Kernel(self.API_CL.program, "gab_find_dzl_sigmoid", 10, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL + SOFTMAX] = new_Kernel(self.API_CL.program, "gab_find_dzl_softmax", 10, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL + IDENTIFY] = new_Kernel(self.API_CL.program, "gab_find_dzl_identify", 10, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, INT /*i0*/);

    self.all_kernels[KERNEL_FIND_DZL_NORMALIZE + ALAN] = new_Kernel(self.API_CL.program, "gab_norm_find_dzl_alan", 13, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, DOUBLE  /*media*/, DOUBLE  /*desvio padrao*/, DOUBLE  /*n*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL_NORMALIZE + TANH] = new_Kernel(self.API_CL.program, "gab_norm_find_dzl_alan", 13, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, DOUBLE  /*media*/, DOUBLE  /*desvio padrao*/, DOUBLE  /*n*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL_NORMALIZE + RELU] = new_Kernel(self.API_CL.program, "gab_norm_find_dzl_alan", 13, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, DOUBLE  /*media*/, DOUBLE  /*desvio padrao*/, DOUBLE  /*n*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL_NORMALIZE + SIGMOID] = new_Kernel(self.API_CL.program, "gab_norm_find_dzl_alan", 13, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, DOUBLE  /*media*/, DOUBLE  /*desvio padrao*/, DOUBLE  /*n*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL_NORMALIZE + IDENTIFY] = new_Kernel(self.API_CL.program, "gab_norm_find_dzl_alan", 13, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, DOUBLE  /*media*/, DOUBLE  /*desvio padrao*/, DOUBLE  /*n*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_DZL_NORMALIZE + SOFTMAX] = new_Kernel(self.API_CL.program, "gab_norm_find_dzl_alan", 13, VOID_P /*dwl_up*/, VOID_P  /*wl_up*/, VOID_P /*dzl_up*/, INT /*m_dzl_up*/, VOID_P /*zl*/, VOID_P /*al*/, VOID_P /*dzl*/, VOID_P /*bl*/, DOUBLE  /*hit*/, DOUBLE  /*media*/, DOUBLE  /*desvio padrao*/, DOUBLE  /*n*/, INT /*i0*/);

    self.all_kernels[KERNEL_FIND_DWL] = new_Kernel(self.API_CL.program, "gab_find_dwl", 5, VOID_P /*dzl*/, VOID_P  /*al_down*/, INT /*mal_down*/, VOID_P /*dwl_up*/, INT /*i0*/);
    self.all_kernels[KERNEL_FIND_AND_UPDATE_DWL] = new_Kernel(self.API_CL.program, "gab_find_dwl_and_update", 7, VOID_P /*wl*/, VOID_P /*dzl*/, INT  /*al_down*/, VOID_P /*mal_down*/, VOID_P /*dwl_up*/, DOUBLE /*hitlearn*/, INT /*i0*/);


    *err = error;
    if (error) {
        fprintf(stderr, "error enquanto criava RNN %d\n", error);
        DNN_release(&self);
    }

    return self;
}

Layer new_Layer(cl_context context, int inp, int out, int func_id, int normalize, int islastorfirst, int *erro) {
    Layer l = {0};
    int error = 0;
    l.funcao_de_ativacao = func_id;
    if (inp != 0) {
        l.w = new_Mat(context, out, inp, &error);
        PER(error, "falha ao criar matriz w")
        l.b = new_Mat(context, out, 1, &error);
        PER(error, "falha ao criar matriz b")
        l.z = new_Mat(context, out, 1, &error);
        PER(error, "falha ao criar matriz z")
        l.dw = new_Mat(context, out, inp, &error);
        PER(error, "falha ao criar matriz dw")
        l.dz = new_Mat(context, out, 1, &error);
        PER(error, "falha ao criar matriz dz")
        if (normalize) {
            l.normalize = 1;
            l.forward = forward_normalize;
            l.backward = backward_normalize;
            if (islastorfirst == -1) {
                l.backward = backward_first_layer_normalize;
            }
        } else {
            l.forward = forward;
            l.backward = backward;
            if (islastorfirst == -1) {
                l.backward = backward_first_layer;
            }
        }
        if (islastorfirst == 1) {
            l.backward = backward_last_layer;
        }

    }
    l.a = new_Mat(context, out, 1, &error);
    PER(error, "falha ao criar matriz a")

    if (error)
        *erro = error;
    return l;
}

void Layer_release(Layer *l) {
    Mat_release(&l->w);
    Mat_release(&l->a);
    Mat_release(&l->b);
    Mat_release(&l->z);
    Mat_release(&l->dz);
    Mat_release(&l->dw);
    memset(l, 0, sizeof(Layer));
}

void DNN_release(DNN *self) {
    int i = 0;
    for (; i <= self->L; i++) {
        Layer_release(self->layers + i);
    }
    free(self->layers);
    free(self->n);
    free(self->out);
    for (i = 0; i < NUMERO_TOTAL_DE_KERNEL; i++) {
        Kernel_release(self->all_kernels + i);
    }
    free(self->all_kernels);
    free(self->functions);
    free(self->normalize);
    clReleaseCommandQueue(self->queue);
}

void DNN_randomize(DNN *self, int *err) {
    int i = 1;
    for (; i <= self->L; i++) {
        Layer_randomize(self, self->layers + i);
    }
}

void DNN_setHitlearn(DNN *self, double hl) {
    self->hLearn = hl;
}

void intern_set_seed(Bytes8 seed) {
    LCG_setSeed(&lcg, seed);
}

int DNN_call(DNN *self, double *input) {
    clEnqueueWriteBuffer(self->queue, self->layers->a.v, CL_TRUE, 0, self->layers->a.bytes, input, 0, NULL, NULL);
    int i;
    for (i = 1; i <= self->L; i++)
        self->layers[i].forward(self, &self->layers[i - 1].a, self->layers + i);
    return 0;
}

int DNN_learn(DNN *self, double *trueout) {
    clEnqueueWriteBuffer(self->queue, self->y.v, CL_TRUE, 0, self->y.bytes, trueout, 0, NULL, NULL);
    int i = self->L;
    int error = 0;
    for (; i > 1; i--) {
//        printf("layer %d\n",i);
        error += self->layers[i].backward(self, self->layers + (i - 1), self->layers + i, self->layers + (i + 1));


    }
    return error;
}


#define call_kernel(total, command)\
    id = 0;\
    global =local= total;\
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

int forward(void *pdnn, void *pinp, void *player) {
    DNN *self = (DNN *) pdnn;
    Mat a_ant = *(Mat *) pinp;
    Layer *l = player;
    size_t global = 1, local = 1;
    Kernel call = self->all_kernels[KERNEL_FULL_FEED + l->funcao_de_ativacao];
    int error = 0;
    int id = 0, resto = 0;
    call_kernel(l->w.m, Kernel_putArgs(&call, 7, &l->w.v, &a_ant.v, &a_ant.m, &l->b.v, &l->z.v, &l->a.v, &id);
            error = clEnqueueNDRangeKernel(self->queue, call.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel call")
    )
    if (l->funcao_de_ativacao == SOFTMAX) {
        double sum = kernel_sum(self, l->a);
        Kernel divide = self->all_kernels[KERNEL_DIVIDE_VECTOR];
        call_kernel(l->a.m, Kernel_putArgs(&divide, 4, &l->a.v, &l->a.v, &sum, &id);
                error = clEnqueueNDRangeKernel(self->queue, divide.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                PERR(error, "erro ao chamar kernel call")
        )
    }
    return error;
}

int forward_normalize(void *pdnn, void *pinp, void *player) {
    DNN *self = (DNN *) pdnn;
    Mat a_ant = *(Mat *) pinp;
    Layer *l = player;
    Kernel call = self->all_kernels[KERNEL_WA_B_FEED];
    size_t global = 1, local = 1;
    int error = 0;
    int id = 0, resto = 0;
    // z = wa+b
    call_kernel(l->w.m, Kernel_putArgs(&call, 6, &l->w.v, &a_ant.v, &a_ant.m, &l->b.v, &l->z.v, &id);
            error = clEnqueueNDRangeKernel(self->queue, call.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel call")
    )

    l->media = kernel_media(self, l->z);
    l->desvio_padrao = kernel_desvio_padrao(self, l->z, l->media);

    Kernel normalize = self->all_kernels[KERNEL_NORMALIZE + l->funcao_de_ativacao];

    call_kernel(l->w.m, Kernel_putArgs(&normalize, 5, &l->a.v, &l->z.v, &l->media, &l->desvio_padrao, &id);
            error = clEnqueueNDRangeKernel(self->queue, normalize.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel normalize")
    )
    if (l->funcao_de_ativacao == SOFTMAX) {
        double sum = kernel_sum(self, l->a);
        Kernel divide = self->all_kernels[KERNEL_DIVIDE_VECTOR];
        call_kernel(l->a.m, Kernel_putArgs(&divide, 4, &l->a.v, &l->a.v, &sum, &id);
                error = clEnqueueNDRangeKernel(self->queue, divide.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                PERR(error, "erro ao chamar kernel call")
        )
    }
    return error;
}

int backward_last_layer(void *pdnn, void *pl0, void *pl, void *py) {
    DNN *self = (DNN *) pdnn;
    Layer *l0 = (Layer *) pl0;
    Layer *l = (Layer *) pl;
    Mat y = self->y;
    Kernel learn = self->all_kernels[KERNEL_FIND_LAST_DZL];
    size_t global = 1, local = 1;
    int error = 0;
    int id = 0, resto = 0;
    call_kernel(l->dz.m, Kernel_putArgs(&learn, 6, &l->a.v, &y.v, &l->dz.v, &l->b.v, &self->hLearn, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel call")
    )
    learn = self->all_kernels[KERNEL_FIND_DWL];
    call_kernel(l->dw.m * l->dw.n, Kernel_putArgs(&learn, 5, &l->dz.v, &l0->a.v, &l0->a.m, &l->dw.v, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dwl")
    )
    return error;
}

int backward(void *pdnn, void *pl0, void *pl, void *pl1) {
    DNN *self = (DNN *) pdnn;
    Layer *l0 = (Layer *) pl0;
    Layer *l = (Layer *) pl;
    Layer *l1 = (Layer *) pl1;
    Kernel learn = self->all_kernels[KERNEL_FIND_DZL + l->funcao_de_ativacao];
    size_t global = 1, local = 1;
    int error = 0;
    int id = 0, resto = 0;
    call_kernel(l->dz.m, Kernel_putArgs(&learn, 10, &l1->dw.v, &l1->w.v, &l1->dz.v, &l1->dz.m, &l->z.v, &l->a.v, &l->dz.v, &l->b.v, &self->hLearn, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dzl")
    )

    learn = self->all_kernels[KERNEL_FIND_DWL];
    call_kernel(l->dw.m * l->dw.n, Kernel_putArgs(&learn, 5, &l->dz.v, &l0->a.v, &l0->a.m, &l->dw.v, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dwl")
    )
    return error;
}

int backward_first_layer(void *pdnn, void *pl0, void *pl, void *pl1) {
    DNN *self = (DNN *) pdnn;
    Layer *l0 = (Layer *) pl0;
    Layer *l = (Layer *) pl;
    Layer *l1 = (Layer *) pl1;
    Kernel learn = self->all_kernels[KERNEL_FIND_DZL + l->funcao_de_ativacao];
    size_t global = 1, local = 1;
    int error = 0;
    int id = 0, resto = 0;
    call_kernel(l->dz.m, Kernel_putArgs(&learn, 10, &l1->dw.v, &l1->w.v, &l1->dz.v, &l1->dz.m, &l->z.v, &l->a.v, &l->dz.v, &l->b.v, &self->hLearn, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dzl")
    )
    learn = self->all_kernels[KERNEL_FIND_AND_UPDATE_DWL];
    call_kernel(l->dw.m * l->dw.n, Kernel_putArgs(&learn, 7, &l->w.v, &l->dz.v, &l0->a.v, &l0->a.m, &l->dw.v, &self->hLearn, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dwl")
    )
    return error;
}


int backward_normalize(void *pdnn, void *pl0, void *pl, void *pl1) {
    DNN *self = (DNN *) pdnn;
    Layer *l0 = (Layer *) pl0;
    Layer *l = (Layer *) pl;
    Layer *l1 = (Layer *) pl1;
    Kernel learn = self->all_kernels[KERNEL_FIND_DZL_NORMALIZE + l->funcao_de_ativacao];
    size_t global = 1, local = 1;
    int error = 0;
    int id = 0, resto = 0;
    double n = l->z.m;
    call_kernel(l->dz.m, Kernel_putArgs(&learn, 10, &l1->dw.v, &l1->w.v, &l1->dz.v, &l1->dz.m, &l->z.v, &l->a.v, &l->dz.v, &l->b.v, &self->hLearn, &l->media, &l->desvio_padrao, &n, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dzl")
    )
    learn = self->all_kernels[KERNEL_FIND_DWL];
    call_kernel(l->dw.m * l->dw.n, Kernel_putArgs(&learn, 5, &l->dz.v, &l0->a.v, &l0->a.m, &l->dw.v, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dwl")
    )
    return error;
}

int backward_first_layer_normalize(void *pdnn, void *pl0, void *pl, void *pl1) {
    DNN *self = (DNN *) pdnn;
    Layer *l0 = (Layer *) pl0;
    Layer *l = (Layer *) pl;
    Layer *l1 = (Layer *) pl1;
    Kernel learn = self->all_kernels[KERNEL_FIND_DZL + l->funcao_de_ativacao];
    size_t global = 1, local = 1;
    int error = 0;
    int id = 0, resto = 0;
    double n = l->z.m;
    call_kernel(l->dz.m, Kernel_putArgs(&learn, 10, &l1->dw.v, &l1->w.v, &l1->dz.v, &l1->dz.m, &l->z.v, &l->a.v, &l->dz.v, &l->b.v, &self->hLearn, &l->media, &l->desvio_padrao, &n, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dzl")
    )
    learn = self->all_kernels[KERNEL_FIND_AND_UPDATE_DWL];
    call_kernel(l->dw.m * l->dw.n, Kernel_putArgs(&learn, 7, &l->w.v, &l->dz.v, &l0->a.v, &l0->a.m, &l->dw.v, &self->hLearn, &id);
            error = clEnqueueNDRangeKernel(self->queue, learn.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            PERR(error, "erro ao chamar kernel find dwl")
    )
    return error;
}


double kernel_sum(DNN *self, Mat x) {
    double s = 0;
    Mat tmp = new_Mat(self->API_CL.context, 1, 1, NULL);
    size_t global = 1, local = 1;
    Kernel sum = self->all_kernels[KERNEL_SUM];
    int len = x.m * x.n;
    int error = 0;
    Kernel_putArgs(&sum, 3, &tmp.v, &x.v, &len);
    error = clEnqueueNDRangeKernel(self->queue, sum.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    PERR(error, "erro ao chamar kernel sum")
    error = clEnqueueReadBuffer(self->queue, tmp.v, CL_TRUE, 0, tmp.bytes, &s, 0, NULL, NULL);
    PERR(error, "error on read buffer ")
    Mat_release(&tmp);
    return s;
}

double kernel_media(DNN *self, Mat x) {
    double s = 0;
    Mat tmp = new_Mat(self->API_CL.context, 1, 1, NULL);
    size_t global = 1, local = 1;
    Kernel sum = self->all_kernels[KERNEL_SUM];
    int len = x.m * x.n;
    int error = 0;
    Kernel_putArgs(&sum, 3, &tmp.v, &x.v, &len);
    error = clEnqueueNDRangeKernel(self->queue, sum.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    PERR(error, "erro ao chamar kernel sum")
    error = clEnqueueReadBuffer(self->queue, tmp.v, CL_TRUE, 0, tmp.bytes, &s, 0, NULL, NULL);
    PERR(error, "error on read buffer ")
    Mat_release(&tmp);
    return s / (len);
}

double kernel_desvio_padrao(DNN *self, Mat x, double media) {
    double s = 0;
    Mat tmp = new_Mat(self->API_CL.context, 1, 1, NULL);
    size_t global = 1, local = 1;
    Kernel sum = self->all_kernels[KERNEL_STD];
    int len = x.m * x.n;
    int error = 0;
    Kernel_putArgs(&sum, 4, &tmp.v, &x.v, &media, &len);
    error = clEnqueueNDRangeKernel(self->queue, sum.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    PERR(error, "erro ao chamar kernel sum")
    error = clEnqueueReadBuffer(self->queue, tmp.v, CL_TRUE, 0, tmp.bytes, &s, 0, NULL, NULL);
    PERR(error, "error on read buffer ")
    Mat_release(&tmp);
    return s;
}

void Layer_randomize(DNN *self, Layer *l) {
    double epsilon = 1.0;// sqrt(6.0 / (l->w.m + l->w.m + 1));
    double *w = (double *) calloc(l->w.bytes, 1);
    int i = l->w.m * l->w.n - 1;
    for (; i >= 0; i--)w[i] = rando(epsilon, -epsilon);
    clEnqueueWriteBuffer(self->queue, l->w.v, CL_TRUE, 0, l->w.bytes, w, 0, NULL, NULL);
    i = l->b.m * l->b.n - 1;
    for (; i >= 0; i--)w[i] = rando(epsilon, -epsilon);
    clEnqueueWriteBuffer(self->queue, l->b.v, CL_TRUE, 0, l->b.bytes, w, 0, NULL, NULL);
    free(w);
}

double rando(double max, double min) {
    max = max - min;
    return (LCG_randD(&lcg)*0.7 +0.3) * max + min;
}


#define CABECALHO "NEURAL NETWORK\n"\
                  "LENGTH OF ARCHITECTURE = 4 BYTES\n"\
                  "ARQUITETURE = 4 * LENGTH_ARCHITECTURE BYTES\n"\
                  "FUNCTION ID = LENGTH_ARCHITECTURE - 1 BYTES\n"\
                  "HITE LEARN = 8 BYTES\n"\
                  "WEIGHTS WITH DIMENSION DEFINED BY ARCHITECTURE\n"

/**
 * Salva em binario os dados da rede
 * os primeiros 210 BYTES sao de cabecalho
 * @param self
 * @param file_name
 * @return
 */
int saveDNN(DNN *self, char *file_name) {
    if (!self)return -2;
    int length = self->L + 1;
    int i, j, l;
    FILE *f = fopen(file_name, "wb");
    if (!f)return -1;

    fwrite(CABECALHO, sizeof(char), 210, f);
    fwrite(&length, sizeof(int), 1, f);
    fwrite(self->n, length, sizeof(int), f);
    fwrite(self->functions, length - 1, sizeof(int), f);
    fwrite(self->normalize, length - 1, sizeof(char), f);
    fwrite(&self->hLearn, 1, sizeof(double), f);

    Mat mb, mw;
    double *w = 0, *b = 0;
    for (l = 1; l <= self->L; l++) {
        mb = self->layers[l].b;
        mw = self->layers[l].b;
        b = realloc(b, mb.bytes);
        w = realloc(w, mw.bytes);
        clEnqueueReadBuffer(self->queue, mb.v, CL_TRUE, 0, mb.bytes, b, 0, NULL, NULL);
        clEnqueueReadBuffer(self->queue, mw.v, CL_TRUE, 0, mw.bytes, w, 0, NULL, NULL);
        for (i = 0; i < mw.m; ++i) {
            fwrite(b + i, 1, sizeof(double), f);
            fwrite(w + (i * mw.n), mw.n, sizeof(double), f);
        }
    }
    free(b);
    free(w);
    fclose(f);
    return 0;
}

DNN loadDNN(WrapperCL *wpr,char *file_name) {
    FILE *f = fopen(file_name, "rb");
    DNN d={0};
    char cabecalho[211] = {0};
    int i, j, l;
    int length, *arq,*fid;
    if (!f)return d;
    fread(cabecalho, sizeof(char), 210, f);
    if (strcmp(cabecalho, CABECALHO)) return d;

    fread(&length, sizeof(int), 1, f);
    arq = calloc(length, sizeof(int));
    fid = calloc(length-1, sizeof(int));

    fread(arq, length, sizeof(int), f);
    fread(fid, length - 1, sizeof(int), f);
    d = new_DNN(wpr,arq, length,fid,NULL,.1,NULL);
    fread(&d.hLearn, 1, sizeof(double), f);
    free(arq);
    free(fid);
    Mat mb, mw;
    double *w = 0, *b = 0;
    for (l = 1; l <= d.L; l++) {
        mb = d.layers[l].b;
        mw = d.layers[l].b;
        b = realloc(b, mb.bytes);
        w = realloc(w, mw.bytes);
        for (i = 0; i < mw.m; ++i) {
            fread(b+ i, 1, sizeof(double), f);
            fread(w + (i * mw.n), mw.n, sizeof(double), f);
        }
        clEnqueueWriteBuffer(d.queue, mb.v, CL_TRUE, 0, mb.bytes,b, 0, NULL, NULL);
        clEnqueueWriteBuffer(d.queue, mw.v, CL_TRUE, 0, mw.bytes,w, 0, NULL, NULL);
    }
    fclose(f);
    return d;
}

int DNN_getA(DNN *self, int l, double *det) {
    if(l<0||l>self->L)return -1;
    clEnqueueReadBuffer(self->queue, self->layers[l].a.v, CL_TRUE, 0, self->layers[l].a.bytes, det, 0, NULL, NULL);
    return 0;
}

void gab_set_max(int mw) {
    max_works = mw;
}







