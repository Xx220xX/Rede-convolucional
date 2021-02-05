//
// Created by Xx220xX on 05/02/2021.
//

#ifndef CNN_GPU_TESTESCNN_H
#define CNN_GPU_TESTESCNN_H
#include "../src/cnn.h"
void getentrada(double *input, int tx, int ty, int tz) {
    double val;

    typetensor t = {0, 0, tx, ty, tz};
    Tensor en = &t;
    for (int x = 0; x < tx; ++x) {
        for (int y = 0; y < ty; ++y) {
            for (int z = 0; z < tz; ++z) {
                val = (sin(x * 2 * M_PI / tx) + sin(y * 2 * M_PI / ty) + sin(z * 2 * M_PI / tz)) / 3.0;
                input[TensorMap(en, x, y, z)] = val;
            }
        }
    }
}

void printTensor(cl_command_queue queue, Kernel *k, Tensor t, int ofset) {
    Kernel_putArgs(k, 5, &t->data, &t->x, &t->y, &t->z, &ofset);
    size_t global = 1, local = 1;
    int error = clEnqueueNDRangeKernel(queue, k->kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(queue);
}

void generateGrad(cl_command_queue queue, Tensor saida, Tensor grad) {
    double *s = (double *) calloc(saida->bytes, 1);
    double *g = (double *) calloc(saida->bytes, 1);
    clEnqueueReadBuffer(queue, saida->data, CL_TRUE, 0, saida->bytes, s, 0, NULL, NULL);
    for (int x = 0; x < saida->x; ++x) {
        for (int y = 0; y < saida->y; ++y) {
            for (int z = 0; z < saida->z; ++z) {
                g[TensorMap(grad, x, y, z)] = s[TensorMap(saida, x, y, z)] * (double) rand() / ((double) RAND_MAX);
            }
        }
    }
    clEnqueueWriteBuffer(queue, grad->data, CL_TRUE, 0, grad->bytes, g, 0, NULL, NULL);
    free(s);
    free(g);
}

int putMen(cl_command_queue queue, Tensor t, double *data) {
    int error = clEnqueueWriteBuffer(queue, t->data, CL_TRUE, 0, t->bytes, data, 0, NULL, NULL);
    clFinish(queue);
    return error;
}

int testeConv() {
    srand(1);
    Params p = {0.1, 0.99, 0.5};
    WrapperCL cl;
    WrapperCL_initbyFile(&cl, "../kernels/gpu_functions.cl");
    setmaxWorks(cl.maxworks);
    Kernel printT = new_Kernel(cl.program, "printTensor", 5, VOID_P, INT, INT, INT, INT);
    Cnn c = createCnn(&cl, p, 5, 5, 3);
    CnnAddConvLayer(c, 1, 3, 2);
    double input[5 * 5 * 3];
    getentrada(input, 5, 5, 3);

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

    Tensor grad = newTensor(c->cl->context, c->camadas[0]->saida->x, c->camadas[0]->saida->y, c->camadas[0]->saida->z, &c->error);
    generateGrad(c->queue, c->camadas[0]->saida, grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    c->camadas[0]->calc_grads(c->camadas[0], grad);
    printTensor(c->queue, &printT, c->camadas[0]->gradsEntrada, 0);

    releaseCnn(&c);
    releaseTensor(&grad);
    WrapperCL_release(&cl);
    Kernel_release(&printT);
    return 0;
}

int testeFullconnect() {
    srand(1);
    Params p = {0.1, 0.99, 0.5};
    WrapperCL cl;
    WrapperCL_initbyFile(&cl, "../kernels/gpu_functions.cl");
    setmaxWorks(cl.maxworks);
    Kernel printT = new_Kernel(cl.program, "printTensor", 5, VOID_P, INT, INT, INT, INT);
    Cnn c = createCnn(&cl, p, 5, 5, 3);
    CnnAddFullConnectLayer(c, 8,  FSIGMOIG);
//    printf("pesos\n");
//    printTensor(c->queue,&printT,((CamadaFullConnect)c->camadas[0])->pesos,0);

    double input[5 * 5 * 3];
    getentrada(input, 5, 5, 3);

    printf("Teste fulconnect\n");
    printf("ativa:\n");
    printf("Tensor de saida\n");
    CnnCall(c, input);
    printTensor(c->queue, &printT, c->camadas[0]->saida, 0);

    Tensor grad = newTensor(c->cl->context, c->camadas[0]->saida->x, c->camadas[0]->saida->y, c->camadas[0]->saida->z, &c->error);
    generateGrad(c->queue, c->camadas[0]->saida, grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    c->camadas[0]->calc_grads(c->camadas[0], grad);
    printTensor(c->queue, &printT, c->camadas[0]->gradsEntrada, 0);

    releaseTensor(&grad);

    releaseCnn(&c);
    WrapperCL_release(&cl);
    Kernel_release(&printT);
    return 0;
}

int testePool() {
    srand(1);
    Params p = {0.1, 0.99, 0.5};
    WrapperCL cl;
    WrapperCL_initbyFile(&cl, "../kernels/gpu_functions_relu.cl");
    setmaxWorks(cl.maxworks);
    Kernel printT = new_Kernel(cl.program, "printTensor", 5, VOID_P, INT, INT, INT, INT);
    Cnn c = createCnn(&cl, p, 5, 5, 3);
    CnnAddPoolLayer(c, 1, 3);
//    printf("pesos\n");
//    printTensor(c->queue,&printT,((CamadaFullConnect)c->camadas[0])->pesos,0);

    double input[5 * 5 * 3];
    getentrada(input, 5, 5, 3);

    printf("Teste pool\n");
    printf("ativa:\n");
    printf("Tensor de saida\n");
    CnnCall(c, input);
    printTensor(c->queue, &printT, c->camadas[0]->saida, 0);

    Tensor grad = newTensor(c->cl->context, c->camadas[0]->saida->x, c->camadas[0]->saida->y, c->camadas[0]->saida->z, &c->error);
    generateGrad(c->queue, c->camadas[0]->saida, grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    printTensor(c->queue, &printT, grad, 0);
    printf("------------CALCULA GRAD-------------\nGRADIENTE entrada\n");
    c->camadas[0]->calc_grads(c->camadas[0], grad);
    printTensor(c->queue, &printT, c->camadas[0]->gradsEntrada, 0);

    releaseTensor(&grad);

    releaseCnn(&c);
    WrapperCL_release(&cl);
    Kernel_release(&printT);
    return 0;
}

int testeRelu() {
    srand(1);
    Params p = {0.1, 0.99, 0.5};
    WrapperCL cl;
    WrapperCL_initbyFile(&cl, "../kernels/gpu_functions_relu.cl");
    setmaxWorks(cl.maxworks);
    Kernel printT = new_Kernel(cl.program, "printTensor", 5, VOID_P, INT, INT, INT, INT);
    Cnn c = createCnn(&cl, p, 5, 5, 3);
    CnnAddReluLayer(c);

    double input[5 * 5 * 3];
    getentrada(input, 5 , 5 , 3);

    printf("Teste pool\n");
    printf("ativa:\n");
    printf("Tensor de saida\n");
    CnnCall(c, input);
    printTensor(c->queue, &printT, c->camadas[0]->saida, 0);

    Tensor grad = newTensor(c->cl->context, c->camadas[0]->saida->x, c->camadas[0]->saida->y, c->camadas[0]->saida->z, &c->error);
    generateGrad(c->queue, c->camadas[0]->saida, grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    printTensor(c->queue, &printT,grad, 0);
    printf("------------CALCULA GRAD-------------\nGRADIENTE entrada\n");
    c->camadas[0]->calc_grads(c->camadas[0], grad);
    printTensor(c->queue, &printT, c->camadas[0]->gradsEntrada, 0);

    releaseTensor(&grad);

    releaseCnn(&c);
    WrapperCL_release(&cl);
    Kernel_release(&printT);
    return 0;
}

#endif //CNN_GPU_TESTESCNN_H
