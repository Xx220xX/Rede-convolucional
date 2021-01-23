//
// Created by Xx220xX on 10/05/2020.
//
#ifndef CL_TESTE_KERNEL_SRC_H
#define CL_TESTE_KERNEL_SRC_H

double ativa(double x) {
    return tanh(x);
}

double df_ativa(double x) {
    double tmp = cosh(x);
    return 1.0 / (tmp * tmp);
}

__kernel void gab_call(
        __global double *w,
        __global double *a_donw, int m_a_donw,
        __global double *b,
        __global double *z,
        __global double *a,
        int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = ativa(tmp);
}

__kernel void gab_sub(__global double *aL, __global double *y, __global double *dzL, __global double *bL, double hit, int i0) {
    int i = get_global_id(0) + i0;
    dzL[i] = aL[i] - y[i];
    bL[i] = bL[i] - hit * dzL[i];
}

__kernel void gab_mult_w_a(__global double *dzl, __global double *al_down, int ma_down, __global double *dwl, int i0) {
    int i = get_global_id(0) + i0;
    int j = i % ma_down;
    i = i / ma_down;
    dwl[i * ma_down + j] = dzl[i] * al_down[j];
}

__kernel void
gab_hidem_dz(__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *dzl,
             __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
    }
    tmp = tmp * df_ativa(zl[i]);
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_update_w(__global double *wl, __global double *dwl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    wl[i] = wl[i] - hit * dwl[i];
}

#endif //CL_TESTE_KERNEL_SRC_H
