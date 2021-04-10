#pragma once

#include "ponto_t.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>
struct tensor_t {
    T *dados;

    tdsize tamanho;

    tensor_t(int _x, int _y, int _z) {
        dados = new T[_x * _y * _z];
        tamanho.x = _x;
        tamanho.y = _y;
        tamanho.z = _z;
    }

    tensor_t(const tensor_t &outro) {
        dados = new T[outro.tamanho.x * outro.tamanho.y * outro.tamanho.z];
        memcpy(
                this->dados,
                outro.dados,
                outro.tamanho.x * outro.tamanho.y * outro.tamanho.z * sizeof(T)
        );
        this->tamanho = outro.tamanho;
    }

    // soma de tensores
    tensor_t<T> operator+(tensor_t<T> &outro) {
        tensor_t<T> clone(*this);
        for (int i = 0; i < outro.tamanho.x * outro.tamanho.y * outro.tamanho.z; i++)
            clone.dados[i] += outro.dados[i];
        return clone;
    }

    // subtracao de tensores
    tensor_t<T> operator-(tensor_t<T> &outro) {
        tensor_t<T> clone(*this);
        for (int i = 0; i < outro.tamanho.x * outro.tamanho.y * outro.tamanho.z; i++)
            clone.dados[i] -= outro.dados[i];
        return clone;
    }

    T &operator()(int _x, int _y, int _z) {
        return this->get(_x, _y, _z);
    }

    T &get(int _x, int _y, int _z) {
        assert(_x >= 0 && _y >= 0 && _z >= 0);
        assert(_x < tamanho.x && _y < tamanho.y && _z < tamanho.z);

        return dados[
                _z * (tamanho.x * tamanho.y) +
                _y * (tamanho.x) +
                _x
        ];
    }

    void copy_from(std::vector<std::vector<std::vector<T>>> dados) {
        int z = dados.size();
        int y = dados[0].size();
        int x = dados[0][0].size();

        for (int i = 0; i < x; i++)
            for (int j = 0; j < y; j++)
                for (int k = 0; k < z; k++)
                    get(i, j, k) = dados[k][j][i];
    }

    ~tensor_t() {
        delete[] dados;
    }
};

static void print_tensor(tensor_t<float> &dados) {
    int mx = dados.tamanho.x;
    int my = dados.tamanho.y;
    int mz = dados.tamanho.z;

    for (int z = 0; z < mz; z++) {
        printf("[Dim%d]\n", z);
        for (int x = 0; x < mx; x++) {
            for (int y = 0; y < my; y++) {

                printf("%.4f \t", (float) dados.get(x, y, z));
            }
            printf("\n");
        }
    }
}

static tensor_t<float> to_tensor(std::vector<std::vector<std::vector<float>>> dados) {
    int z = dados.size();
    int y = dados[0].size();
    int x = dados[0][0].size();


    tensor_t<float> t(x, y, z);

    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            for (int k = 0; k < z; k++)
                t(i, j, k) = dados[k][j][i];
    return t;
}
