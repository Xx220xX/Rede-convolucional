//
// Created by Xx220xX on 05/05/2020.
//
/**
 *    HOW COMPILE THIS?
 *      win 64bit
           gcc -m64 -c GabrielaGPUPADA.c -o gabriela64.o
           gcc -m64  -shared -o gabriela64.dll gabriela64.o
 *      win 32bit
           gcc -m32 -c gabriela.c -o gabriela32.o
           gcc -m32  -shared -o gabriela64.dll gabriela32.o
 *
 *
 */
#ifdef GABRIELAGPUPADA_H
#define GABRIELAGPUPADA_H

#define GPU_CL

#include "MatrixGPUPADA.h"
#include <stdio.h>

#ifdef GPU_CL
#include "config_gpu_access.h"
#define FUNC_ID_TANH 0
#define FUNC_ID_DFTANH 1
#define FUNC_ID_RELU 2
#define FUNC_ID_DFRELU 3
#define FUNC_ID_SIGMOID 8
#define FUNC_ID_DFSIGMOID 9
#define FUNC_ID_ALAN 16
#define FUNC_ID_DFALAN 17
#define FLAG_DIF 1
#else
#include "Optime_GPU.cl"
#include "config_cpu_access.h"
#include <math.h>
#endif
#ifdef __cplusdcplus
extern "C" {
#endif

typedef struct {
  int L;      // last layer
  int arq_0;  // neuronios on first layer
  int arq_o;  // neuronios on output layer
  Mat *a;     // after activate
  Mat *z;     // sum(Wij*Ali)
  Mat *w, *b; //  weight and bias
} GAB;
typedef struct {
  int len, *p;
} Vector;
static int activation_type = FUNC_ID_TANH;

#define __cplusdcplus

#include "GabrielaGPUPADA.h"

#ifdef __cplusdcplus
extern "C" {
#endif

#ifdef DNN__DEBUG
void pMat(const char *n, Mat *m) {
  int i, j;
  printf("%s : %dx%d", n, m->m, m->n);

  for (i = 0; i < m->m; i++) {
    printf("\n\t|");
    for (j = 0; j < m->n; j++) {
      printf("%.4lf ", m->v[i * m->n + j]);
    }
  }
  printf("\n");
}

#endif // DNN__DEBUG

int setFuncActivation(int id) {
  switch (id) {
  case FUNC_ID_TANH:
  case FUNC_ID_RELU:
  case FUNC_ID_SIGMOID:
  case FUNC_ID_ALAN:
    activation_type = id;
    return 0;
  default:
    return -1;
  }
}

int getFuncActivation() { return activation_type; }

int call(GAB *g, double *input) {
  Mat inp = {g->arq_0, 1, input}; // inp^T
  Mcpy(&g->a[0], &inp);
  int l;
#ifdef DNN__DEBUG
  char nome[100];
#endif
  for (l = 1; l <= g->L; l++) {
    // z[l] = w[l] * a[l-1] + b[l]
    // a[l] = ativa(z[l])
    API_GPU_CALL(g->a[l], g->z[l], g->w[l], g->a[l - 1], g->b[l],
                 activation_type);
#ifdef DNN__DEBUG
    snprintf(nome, 100, "z%d", l);
    pMat(nome, &g->z[l]);
    snprintf(nome, 100, "a%d", l);
    pMat(nome, &g->a[l]);
#endif
  }
  return 0;
}

int aprende(GAB *g, double *output, double hitLearn) {
  int l;
#ifdef DNN__DEBUG
  printf("_______aprende_________\n\n");
#endif

  Mat out = {g->arq_o, 1, output};
  // calcular derivadas
  Mat *dz = (Mat *)calloc(g->L + 1, sizeof(Mat));
  Mat *dW = (Mat *)calloc(g->L + 1, sizeof(Mat));
  // ultima camada
  dz[g->L] = newMat(out.m, out.n);
  dW[g->L] = newMat(dz[g->L].m, g->a[g->L - 1].m);
  // dz[L] = a[L] - out
  // dW[L] = dz[L] * a[L-1]^t
#ifdef DNN__DEBUG
  pMat("W[L]", &g->w[g->L]);
  pMat("b[L]", &g->b[g->L]);
  printf("steps\n");
#endif
  API_GPU_LAST_LAYER(dz[g->L], g->a[g->L], output, dW[g->L], g->a[g->L - 1],
                     g->w[g->L], g->b[g->L]);
#ifdef DNN__DEBUG

  printf("dz[L] = a[L] - out\n");
  pMat("a[L]", &g->a[g->L]);
  pMat("out", &out);
  pMat("dz[L]", &dz[g->L]);

  printf("dW[L] = dz[L] * a[l-1]^t\n");
  pMat("dz[L]", &dz[g->L]);
  pMat("a[L-1]", &g->a[g->L - 1]);
  pMat("dW[L]", &dW[g->L]);
  printf("results \n");
  pMat("W[L]", &g->w[g->L]);
  pMat("b[L]", &g->b[g->L]);

  printf("_______internas_________\n\n");
#endif
  // Camadas internas
  for (l = g->L - 1; l > 0; l--) {

    dz[l] = newMat(g->w[l + 1].n, g->z[l].n);
    dW[l] = newMat(dz[l].m, g->a[l - 1].m);
    // dz[l] = w[l+1]^t * (dz[l+1]* dif(z[l]))
    // dw[l] = dz[l] * a[l-1]^t
    API_GPU_HIDDEN_LAYER(dz[l], g->w[l + 1], dz[l + 1], g->z[l], g->a[l - 1],
                         dW[l], g->w[l], g->b[l], activation_type);

#ifdef DNN__DEBUG
    printf("l = %d\n", l);
    printf("before \n");
    pMat("W[l]", &g->w[l]);
    pMat("b[l]", &g->b[l]);
    printf("steps\n");
    printf("\ndz[l] = (w[l+1]^t * dz[l+1]) x f'(z[l])\n");
    pMat("w[l+1]", &g->w[l + 1]);
    pMat("dz[l+1]", &dz[l + 1]);
    pMat("z[l]", &g->z[l]);
    pMat("dz[l]", &dz[l]);

    printf("\ndW[l] = dz[l] * a[l-1]^t\n");
    pMat("a[l-1]", &g->a[l - 1]);
    pMat("dW[l]", &dW[l]);
    printf("results \n");
    pMat("W[l]", &g->w[l]);
    pMat("b[l]", &g->b[l]);

    printf("________________\n\n");
#endif
  }
  // liberar matrizes usadas nesta funcao
  for (l = 1; l <= g->L; l++) {
    API_GPU_SET_WEIGHT(dz[l], dW[l], g->w[l], g->b[l], hitLearn);
    freeMat(&dz[l]);
    freeMat(&dW[l]);
  }
  freeMat(&dz[0]);
  freeMat(&dW[0]);

  free(dz);
  free(dW);
  //  printf("end  c\n");

  return 0;
}

int save(char *file2save, GAB *g, int n_arq, int *arq) {
  FILE *f = fopen(file2save, "wb");
  int i, l;
  if (!f)
    return -1;
  fwrite(&n_arq, sizeof(int), 1, f);
  fwrite(arq, sizeof(int), n_arq, f);
  for (l = 1; l <= g->L; l++) {
    fwrite(g->w[l].v, sizeof(double), g->w[l].m * g->w[l].n, f);
    fwrite(g->b[l].v, sizeof(double), g->b[l].m * g->b[l].n, f);
  }
  fclose(f);
  return 0;
}

int preLoad(char *file2load, Vector *arquitetura) {
  FILE *f = fopen(file2load, "rb");
  int i, l;
  if (!f)
    return -1;
  fread(&arquitetura->len, sizeof(int), 1, f);
  arquitetura->p = (int *)calloc(arquitetura->len, sizeof(int));
  fread(arquitetura->p, sizeof(int), arquitetura->len, f);
  fclose(f);
  return 0;
}

int load(char *file2load, Vector *arquitetura, GAB *g) {
  FILE *f = fopen(file2load, "rb");
  int l;
  if (!f)
    return -1;
  fread(&arquitetura->len, sizeof(int), 1, f);
  fread(arquitetura->p, sizeof(int), arquitetura->len, f);
  for (l = 1; l <= g->L; l++) {
    fread(g->w[l].v, sizeof(double), g->w[l].m * g->w[l].n, f);
    fread(g->b[l].v, sizeof(double), g->b[l].m * g->b[l].n, f);
  }
  fclose(f);
  free(arquitetura->p);
  return 0;
}

#endif // GABRIELAGPUPADA_H
