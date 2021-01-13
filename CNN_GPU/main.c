#include <stdio.h>
#include "src/cnn.h"

void generateInput(Tensor t) {
    double val;
    for (int i = 0; i < t->tx; ++i) {
        for (int j = 0; j < t->ty; ++j) {
            for (int k = 0; k < t->tz; ++k) {
                val = (sin(i * 2 * M_PI / t->tx) + sin(j * 2 * M_PI / t->ty) + sin(k * 2 * M_PI / t->tz)) / 3.0;
                TensorAT(t, i, j, k) = val;
            }
        }
    }
}
void generategrad(Tensor saida, Tensor grad) {
    for (int x = 0; x < saida->tx; ++x) {
        for (int y = 0; y < saida->ty; ++y) {
            for (int z = 0; z < saida->tz; ++z) {
                TensorAT(grad,x,y,z) = TensorAT(saida,x,y,z) * (double)rand() / ((double) RAND_MAX);
            }
        }
    }

}
void testePool() {
    srand(1);
    Params p = {0.1, 0.6, 0.001};
    Tensor entrada = newTensor(5, 5, 3);
    generateInput(entrada);
    Camada pl = createPool(1, 3, 5, 5, 3, entrada, &p);
    pl->ativa(pl);

    printf("------------ATIVA-------------\nSAIDA\n");
    printTensor(pl->saida);


    Tensor grad = newTensor(pl->saida->tx, pl->saida->ty, pl->saida->tz);
    for (int i = 0; i < pl->saida->tx * pl->saida->ty * pl->saida->tz; i++) {
        grad->data[i] = pl->saida->data[i] * .3;
    }
    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    printTensor(grad);
    pl->calc_grads(pl, grad);

    printf("GRADIENTE ENTRADA\n");
    printTensor(pl->gradsEntrada);
    pl->release(&pl);
    releaseTensor(&entrada);
    releaseTensor(&grad);

}

void testeConv() {
    srand(1);
    Params p = {0.1, 0.6, 0.001};
    Tensor entrada = newTensor(5, 5, 3);
    generateInput(entrada);
    Camada pl = createConv(1, 3, 2, 5, 5, 3, entrada, &p);
    pl->ativa(pl);

    printf("Teste conv\n\nativa:\nTensor de saida\n");
    printTensor(pl->saida);

    Tensor grad = newTensor(pl->saida->tx, pl->saida->ty, pl->saida->tz);
    generategrad(pl->saida,grad);
    printf("------------CALCULA GRAD-------------\n");
    pl->calc_grads(pl, grad);

    printf("GRADIENTE ENTRADA\n");
    printTensor(pl->gradsEntrada);



    pl->release(&pl);
    releaseTensor(&entrada);
    releaseTensor(&grad);
}

int main() {
    testeConv();
    /*Params p = {0.1,0.99,0.5};
    Cnn c = createCnn(p,28,28,3);
    CnnAddConvLayer(c,1,3,8);
    CnnAddReluLayer(c);
    CnnAddPoolLayer(c,2,2);
    CnnAddFullConnectLayer(c,10,SIGMOIG);
    releaseCnn(&c);*/

    return 0;
}
