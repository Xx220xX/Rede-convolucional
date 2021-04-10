__kernel void gab_divide_vector(__global double *ans, __global double *x, double value, int i0) {
    int i = get_global_id(0) + i0;
    ans[i] = x[i] / value;
}

__kernel void gab_desvio_padrao(__global double *ans, __global double *x, double media, int n) {
    double tmp = 0, aux;
    for (int i = 0; i < n; i++) {
        aux = (x[i] - media);
        tmp += aux * aux;
    }
    ans[0] = sqrt(tmp / (double) n);
}

__kernel void gab_sum(__global double *ans, __global double *x, int n) {
    double tmp = 0;
    for (int i = 0; i < n; i++) {
        tmp += x[i];
    }
    ans[0] = tmp;
}
