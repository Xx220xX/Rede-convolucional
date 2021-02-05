#include <iostream>
#include <time.h>
#include "src/cnn.h"

tensor_t<float> getEntrada(int xx, int yy, int zz) {
    tensor_t<float> en(xx, yy, zz);
    double val;
    for (int x = 0; x < en.tamanho.x; ++x) {
        for (int y = 0; y < en.tamanho.y; ++y) {
            for (int z = 0; z < en.tamanho.z; ++z) {
                val = (sin(x * 2 * M_PI / en.tamanho.x) + sin(y * 2 * M_PI / en.tamanho.y) + sin(z * 2 * M_PI / en.tamanho.z)) / 3.0;
                en(x, y, z) = (float) val;
            }
        }
    }
//    printf("\n");
//    for(int i=0;i<xx*yy*zz;i++){
//        printf("%f,",en.dados[i]);
//    }
    return en;
}

void generategrad(tensor_t<float> saida, tensor_t<float> &grad) {
    for (int x = 0; x < saida.tamanho.x; ++x) {
        for (int y = 0; y < saida.tamanho.y; ++y) {
            for (int z = 0; z < saida.tamanho.z; ++z) {
                grad(x, y, z) = saida(x, y, z) * (double) rand() / ((double) RAND_MAX);
            }
        }
    }

}

void testConv() {
    srand(1);
    camada_conv_t *cconv = new camada_conv_t(1, 3, 2, {5, 5, 3});
    tensor_t<float> entrada = getEntrada(5, 5, 3);
    printf("filtros:\n1)\n");
    print_tensor(cconv->filtros[0]);
    printf("2)\n");
    print_tensor(cconv->filtros[1]);
    printf("+++++++++\n");

    cconv->ativa(entrada);
    printf("Teste conv\n\nativa:\nTensor de saida\n");
    print_tensor(cconv->saida);

    tensor_t<float> grad(cconv->saida.tamanho.x, cconv->saida.tamanho.y, cconv->saida.tamanho.z);
    generategrad(cconv->saida, grad);

    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    cconv->calc_grads(grad);
    print_tensor(cconv->grads_entrada);


}

void testePool() {
    srand(1);
    camada_pool_t *cpool = new camada_pool_t(1,3,{5, 5, 3});
    tensor_t<float> entrada = getEntrada(5, 5, 3);
    cpool->ativa(entrada);
    printf("Teste fullconnect\n\nativa:\n");
    printf("Tensor de saida\n");
    print_tensor(cpool->saida);

    tensor_t<float> grad(cpool->saida.tamanho.x, cpool->saida.tamanho.y, cpool->saida.tamanho.z);
    generategrad(cpool->saida, grad);

    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    print_tensor(grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE entrada\n");
    cpool->calc_grads(grad);
    print_tensor(cpool->grads_entrada);



}
void testeRelu() {
    srand(1);
    camada_relu_t *crelu = new camada_relu_t( {5, 5, 3});
    tensor_t<float> entrada = getEntrada(5, 5, 3);
    crelu->ativa(entrada);
    printf("Teste fullconnect\n\nativa:\n");
    printf("Tensor de saida\n");
    print_tensor(crelu->saida);

    tensor_t<float> grad(crelu->saida.tamanho.x, crelu->saida.tamanho.y, crelu->saida.tamanho.z);
    generategrad(crelu->saida, grad);

    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    print_tensor(grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE entrada\n");
    crelu->calc_grads(grad);
    print_tensor(crelu->grads_entrada);



}
int testeFullCOnnect() {
    srand(1);
    camada_fc_t *cfull = new camada_fc_t({5, 5, 3}, 8);
    tensor_t<float> entrada = getEntrada(5, 5, 3);
    cfull->ativa(entrada);
    printf("Teste fullconnect\n\nativa:\n");
//    printf("pesos\n");
//    print_tensor(cfull->pesos);
    printf("Tensor de saida\n");
    print_tensor(cfull->saida);

    tensor_t<float> grad(cfull->saida.tamanho.x, cfull->saida.tamanho.y, cfull->saida.tamanho.z);
    generategrad(cfull->saida, grad);

    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    print_tensor(grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE entrada\n");
    cfull->calc_grads(grad);
    print_tensor(cfull->grads_entrada);

    return 0;
}
int main(){
    testeRelu();
    return 0;
}