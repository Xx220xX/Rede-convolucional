//
// Created by Xx220xX on 12/05/2020.
//

#include <stdio.h>
#include <stdlib.h>
#include "matGPU.h"



Mat new_Mat(cl_context context, int m, int n, int *err) {
    Mat self = {0};
    if (err && *err)return self;
    self.bytes = m * n * sizeof(double);
    cl_int error = 0;
    self.m = m;
    self.n = n;
    self.v = clCreateBuffer(context, CL_MEM_READ_WRITE, self.bytes, NULL, &error);
    if (self.v == NULL) {
        error = -1;
        fprintf(stderr, "A memoria retornada foi NULL\n");
    }

    if (error) {
        fprintf(stderr, "nao foi possivel criar memoria\n");
    }
    if (err) {
        *err = error;
    }
    return self;
}

void Mat_release(Mat *self) {
    if (self->v)clReleaseMemObject(self->v);
    self->m = self->n = 0;
    self->v = NULL;
}


void Mat_print(Mat *self, cl_command_queue queue) {
    double *p = (double *) calloc(self->bytes, 1);
    clEnqueueReadBuffer(queue, self->v, CL_TRUE, 0, self->bytes, p, 0, NULL, NULL);
    printf("%dx%d\n", self->m, self->n);
    for (int i = 0; i < self->m; ++i) {
        for (int j = 0; j < self->n; ++j) {
            printf("%.4lf ", p[i * self->n + j]);
        }
        printf("\n");
    }
    printf("\n");
    free(p);
}
