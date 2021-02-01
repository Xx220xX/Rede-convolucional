#pragma once

#include "camada_t.h"

#pragma pack(push, 1)

struct camada_conv_t {
    tipo_camada tipo = tipo_camada::conv;
    tensor_t<float> grads_entrada;
    tensor_t<float> entrada;
    tensor_t<float> saida;
    std::vector<tensor_t<float>> filtros;
    std::vector<tensor_t<gradiente_t>> grads_filtros;
    uint16_t passo;
    uint16_t tam_filtro;

    /***
       * o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
       * em que:
       * 			S = tamanho da saida
       * 			E = tamanho da entrada
       * 			F = tamanho do filtro
       * 			Pd = preenchimento com zeros
       * 			P = passo
       *
       * nesse codigo nao esta sendo usado preenchimento com zero --> Pd = 0
       */
    camada_conv_t(uint16_t passo, uint16_t tam_filtro, uint16_t num_filtros, tdsize tam_entrada)
            :
            grads_entrada(tam_entrada.x, tam_entrada.y, tam_entrada.z),
            entrada(tam_entrada.x, tam_entrada.y, tam_entrada.z),
            saida(
                    (tam_entrada.x - tam_filtro) / passo + 1,
                    (tam_entrada.y - tam_filtro) / passo + 1,
                    num_filtros
            ) {
        this->passo = passo;
        this->tam_filtro = tam_filtro;
        assert((float(tam_entrada.x - tam_filtro) / passo + 1)
               ==
               ((tam_entrada.x - tam_filtro) / passo + 1));

        assert((float(tam_entrada.y - tam_filtro) / passo + 1)
               ==
               ((tam_entrada.y - tam_filtro) / passo + 1));


        // inicializa os coeficientes de todos os filtros aleatoriamente
        for (int a = 0; a < num_filtros; a++) {
            tensor_t<float> t(tam_filtro, tam_filtro, tam_entrada.z);

            int maxval = tam_filtro * tam_filtro * tam_entrada.z;

            for (int i = 0; i < tam_filtro; i++)
                for (int j = 0; j < tam_filtro; j++)
                    for (int z = 0; z < tam_entrada.z; z++)
                        t(i, j, z) = 1.0f / maxval * rand() / float(RAND_MAX);
            filtros.push_back(t);
        }

        // inicializa os gradientes dos filtros
        for (int i = 0; i < num_filtros; i++) {
            tensor_t<gradiente_t> t(tam_filtro, tam_filtro, tam_entrada.z);
            grads_filtros.push_back(t);
        }

    }

    // funcao que mapeia um ponto de coordenadas da saida para coordenadas da entrada
    ponto_t mapeia_saida_entrada(ponto_t saida, int z) {
        saida.x *= passo;
        saida.y *= passo;
        saida.z = z;
        return saida;
    }

    // faixa de variacao das coordenadas
    struct range_t {
        int min_x, min_y, min_z;
        int max_x, max_y, max_z;
    };


    // normaliza a faixa de variacao das coordenadas
    int normaliza_range(float f, int max, bool lim_min) {
        if (f <= 0)
            return 0;
        max -= 1;
        if (f >= max)
            return max;

        if (lim_min)
            return ceil(f);
        else
            return floor(f);
    }

    // mapeia um ponto de coordenadas da entrada para coordenadas da saida
    range_t mapeia_entrada_saida(int x, int y) {
        float a = x;
        float b = y;
        return
                {
                        normaliza_range((a - tam_filtro + 1) / passo, saida.tamanho.x, true),
                        normaliza_range((b - tam_filtro + 1) / passo, saida.tamanho.y, true),
                        0,
                        normaliza_range(a / passo, saida.tamanho.x, false),
                        normaliza_range(b / passo, saida.tamanho.y, false),
                        (int) filtros.size() - 1,
                };
    }

    void ativa(tensor_t<float> &entrada) {
        this->entrada = entrada;
        ativa();
    }


    // ativa a camada realizando a soma de convolucao dos filtros com a entrada
    void ativa() {
        // para cada um dos filtros
        for (int num_filtro = 0; num_filtro < filtros.size(); num_filtro++) {
            // cria um tensor com os coeficientes do filtro sendo utilizado
            tensor_t<float> &filtro = filtros[num_filtro];
            for (int x = 0; x < saida.tamanho.x; x++) {
                for (int y = 0; y < saida.tamanho.y; y++) {
                    // converte as coordenadas da saida para entrada
                    ponto_t mapeado = mapeia_saida_entrada({(uint16_t) x, (uint16_t) y, 0}, 0);
                    // opera a soma de convolucao
                    float sum = 0;
                    for (int i = 0; i < tam_filtro; i++)
                        for (int j = 0; j < tam_filtro; j++)
                            for (int z = 0; z < entrada.tamanho.z; z++) {
                                float f = filtro(i, j, z);
                                float v = entrada(mapeado.x + i, mapeado.y + j, z);
                                sum += f * v;
                            }

                    // coloca a soma de convolucao na saida da camada
                    saida(x, y, num_filtro) = sum;
                }
            }
        }
    }

    void corrige_pesos() {
        for (int a = 0; a < filtros.size(); a++)
            for (int i = 0; i < tam_filtro; i++)
                for (int j = 0; j < tam_filtro; j++)
                    for (int z = 0; z < entrada.tamanho.z; z++) {
                        float &w = filtros[a].get(i, j, z);
                        gradiente_t &grad = grads_filtros[a].get(i, j, z);
                        w = atualiza_peso(w, grad);
                        atualiza_gradiente(grad);
                    }
    }

    void calc_grads(tensor_t<float> &grad_prox_camada) {

        for (int k = 0; k < grads_filtros.size(); k++) {
            for (int i = 0; i < tam_filtro; i++)
                for (int j = 0; j < tam_filtro; j++)
                    for (int z = 0; z < entrada.tamanho.z; z++)
                        grads_filtros[k].get(i, j, z).grad = 0;
        }

        for (int x = 0; x < entrada.tamanho.x; x++) {
            for (int y = 0; y < entrada.tamanho.y; y++) {
                range_t rn = mapeia_entrada_saida(x, y);
                for (int z = 0; z < entrada.tamanho.z; z++) {
                    float soma_erro = 0;
                    for (int i = rn.min_x; i <= rn.max_x; i++) {
                        int minx = i * passo;
                        for (int j = rn.min_y; j <= rn.max_y; j++) {
                            int miny = j * passo;
                            for (int k = rn.min_z; k <= rn.max_z; k++) {
                                float peso_aplicado = filtros[k].get(x - minx, y - miny, z);
                                soma_erro += peso_aplicado * grad_prox_camada(i, j, k);
                                grads_filtros[k].get(x - minx, y - miny, z).grad += entrada(x, y, z) * grad_prox_camada(i, j, k);
                            }
                        }
                    }
                    grads_entrada(x, y, z) = soma_erro;
                }
            }
        }
    }
};

#pragma pack(pop)
