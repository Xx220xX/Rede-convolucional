#include <stdio.h>
#include "src/cnn.h"

void getentrada(double *input, int len) {
    double v[] = {0.000000, 0.317019, 0.195928, -0.195928, -0.317019, 0.317019, 0.634038, 0.512947, 0.121090, -0.000000, 0.195928, 0.512947, 0.391857,
                  0.000000, -0.121090, -0.195928, 0.121090, 0.000000, -0.391857, -0.512947, -0.317019, -0.000000, -0.121090, -0.512947, -0.634038,
                  0.288675, 0.605694, 0.484604, 0.092747, -0.028344, 0.605694, 0.922713, 0.801622, 0.409766, 0.288675, 0.484604, 0.801622, 0.680532,
                  0.288675, 0.167585, 0.092747, 0.409766, 0.288675, -0.103182, -0.224272, -0.028344, 0.288675,
                  0.167585, -0.224272, -0.345363, -0.288675, 0.028344, -0.092747, -0.484604, -0.605694, 0.028344,
                  0.345363, 0.224272, -0.167585, -0.288675, -0.092747, 0.224272, 0.103182, -0.288675, -0.409766,
                  -0.484604, -0.167585, -0.288675, -0.680532, -0.801622, -0.605694, -0.288675, -0.409766, -0.801622, -0.922713};
    memcpy(input, v, len * sizeof(double));
}

void printTensor(cl_command_queue queue, Kernel *k, Tensor t, int ofset) {
    Kernel_putArgs(k, 5, &t->data, &t->x, &t->y, &t->z, &ofset);
    size_t global = 1, local = 1;
    int error = clEnqueueNDRangeKernel(queue, k->kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(queue);
}
void generateGrad(cl_command_queue queue,Tensor saida,Tensor grad){
    double * s = (double*)calloc(saida->bytes,1);
    double * g = (double*)calloc(saida->bytes,1);
    clEnqueueReadBuffer(queue,saida->data,CL_TRUE,0,saida->bytes,s,0,NULL,NULL);
    for (int x = 0; x < saida->x; ++x) {
        for (int y = 0; y < saida->y; ++y) {
            for (int z = 0; z < saida->z; ++z) {
                g[TensorMap(grad,x, y, z)] = s[TensorMap(saida,x, y, z)] * (double) rand() / ((double) RAND_MAX);
            }
        }
    }
    clEnqueueWriteBuffer(queue,grad->data,CL_TRUE,0,grad->bytes,g,0,NULL,NULL);
    free(s);free(g);
}
int putMen(cl_command_queue queue, Tensor t, double *data) {
    int error = clEnqueueWriteBuffer(queue, t->data, CL_TRUE, 0, t->bytes, data, 0, NULL, NULL);
    clFinish(queue);
    return error;
}

int main() {
    srand(1);
    Params p = {0.1, 0.99, 0.5};
    WrapperCL cl;
    WrapperCL_initbyFile(&cl, "../kernels/gpu_functions.cl");
    setmaxWorks(cl.maxworks);
    Kernel printT = new_Kernel(cl.program, "printTensor", 5, VOID_P, INT, INT, INT, INT);
    Cnn c = createCnn(&cl, p, 5, 5, 3);
    CnnAddConvLayer(c, 1, 3, 2);

    double input[5 * 5 * 3];
    getentrada(input, 5 * 5 * 3);

    printf("filtros:\n1)\n");
    printTensor(c->queue, &printT, ((CamadaConv) c->camadas[0])->filtros, 0);


    printf("2)\n");
    printTensor(c->queue, &printT, ((CamadaConv) c->camadas[0])->filtros, 3 * 3 * 3);
    printf("+++++++++\n");

    printf("Teste Cnv\n");
    printf("ativa:\n");
    printf("Tensor de saida\n");
    CnnCall(c, input);
    printTensor(c->queue, &printT, c->camadas[0]->saida, 0);

    Tensor grad = newTensor(c->cl->context,c->camadas[0]->saida->x,c->camadas[0]->saida->y,c->camadas[0]->saida->z,&c->error);
    generateGrad(c->queue,c->camadas[0]->saida,grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    c->camadas[0]->calc_grads(c->camadas[0],grad);
    printTensor(c->queue, &printT, c->camadas[0]->gradsEntrada, 0);

    releaseCnn(&c);
    releaseTensor(&grad);
    WrapperCL_release(&cl);
    Kernel_release(&printT);
    return 0;
}
