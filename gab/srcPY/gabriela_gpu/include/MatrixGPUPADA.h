//
// Created by Xx220xX on 04/05/2020.
//

#ifndef GAME2D_MATRIXGPUPADA_H
#define GAME2D_MATRIXGPUPADA_H

#include <stdlib.h>


typedef struct {
    int m, n;
    double *v;
} Mat;


/** Copia matrizes ambas devem ter mesmo tamanho e ja instanciadas*/
void Mcpy(Mat *a, Mat *b) {
    int i = 0;
    int l = a->m * a->n;
    for (i = 0; i < l; i++) {
        a->v[i] = b->v[i];
    }
}

Mat newMat(int m, int n) {
    Mat ans = {m, n, 0};
    ans.v = (double *) calloc(ans.n * ans.m, sizeof(double));

    return ans;
}

void freeMat(Mat *m) {
    if (m->v) {
        free(m->v);
        m->v = 0;
    }
}

#endif //GAME2D_MATRIXGPUPADA_H
