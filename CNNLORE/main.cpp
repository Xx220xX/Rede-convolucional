#include <iostream>
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
    camada_pool_t *cconv = new camada_pool_t(1, 3, {5, 5, 3});
    tensor_t<float> entrada = getEntrada(5, 5, 3);
    cconv->ativa(entrada);
    printf("Teste conv\n\nativa:\nTensor de saida\n");
    print_tensor(cconv->saida);

    tensor_t<float> grad(cconv->saida.tamanho.x, cconv->saida.tamanho.y, cconv->saida.tamanho.z);
    generategrad(cconv->saida, grad);

    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    print_tensor(grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE entrada\n");
    cconv->calc_grads(grad);
    print_tensor(cconv->grads_entrada);


}

int main() {
    srand(1);
    camada_fc_t *cconv = new camada_fc_t({5,5,3},8);
    tensor_t<float> entrada = getEntrada(5, 5, 3);
    cconv->ativa(entrada);
    printf("Teste conv\n\nativa:\nTensor de saida\n");
    print_tensor(cconv->saida);

    tensor_t<float> grad(cconv->saida.tamanho.x, cconv->saida.tamanho.y, cconv->saida.tamanho.z);
    generategrad(cconv->saida, grad);

    printf("------------CALCULA GRAD-------------\nGRADIENTE\n");
    print_tensor(grad);
    printf("------------CALCULA GRAD-------------\nGRADIENTE entrada\n");
    cconv->calc_grads(grad);
    print_tensor(cconv->grads_entrada);

    return 0;
}