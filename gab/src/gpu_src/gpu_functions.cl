//
// Created by Xx220xX on 10/05/2020.
//a
#ifndef CL_TESTE_KERNEL_SRC_H
#define CL_TESTE_KERNEL_SRC_H


__kernel void gab_feed(__global double *w, __global double *a_donw, int m_a_donw, __global double *b, __global double *z, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
}

__kernel void gab_normalize_alan(__global double *a, __global double *z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    if (tmp > 1)tmp = log(tmp - 1);
    else if (tmp < 1)tmp = -log(-tmp - 1);
    a[i] = tmp;
}

__kernel void gab_normalize_tanh(__global double *a, __global double *z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = tanh(tmp);
}

__kernel void gab_normalize_relu(__global double *a, __global double *z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = tmp > 0.0 ? tmp : 0.0;
}

__kernel void gab_normalize_sigmoid(__global double *a, __global double *z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = 1.0 / (1.0 + exp(-tmp));
}

__kernel void gab_normalize_identify(__global double *a, __global double *z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = tmp;
}

__kernel void gab_normalize_softmax(__global double *a, __global double *z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = exp(tmp);
}





__kernel void gab_find_last_dzl(__global double *aL, __global double *y, __global double *dzL, __global double *bL, double hit, int i0) {
    int i = get_global_id(0) + i0;
    dzL[i] = aL[i] - y[i];
    bL[i] = bL[i] - hit * dzL[i];
}

__kernel void gab_find_dwl(__global double *dzl, __global double *al_down, int ma_down, __global double *dwl, int i0) {
    int i = get_global_id(0) + i0;
    int j = i % ma_down;
    i = i / ma_down;
    dwl[i * ma_down + j] = dzl[i] * al_down[j];
}

__kernel void gab_find_dwl_and_update(__global double *wl, __global double *dzl, __global double *al_down, int ma_down, __global double *dwl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    int j = i % ma_down;
    i = i / ma_down;
    dwl[i * ma_down + j] = dzl[i] * al_down[j];
    wl[i * ma_down + j] = wl[i * ma_down + j] - hit * dwl[i * ma_down + j];
}

__kernel void gab_hidem_dz(__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *dzl, __global double *bl, double hit, int i0) {
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
