__kernel void gab_find_dzl_alan(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = zl[i];
    if (dif > 1.0)dif = 1.0 / (dif - 1);
    else if (dif < 1.0)dif = 1.0 / (-dif - 1);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_find_dzl_tanh(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = al[i];
    dif = 1 - dif * dif;// 1 - tanh(x)^2
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_find_dzl_tanh(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = al[i];
    dif = 1 - dif * dif;// 1 - tanh(x)^2
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_find_dzl_relu(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 0.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    if (z[i] > 0)dif = 1;
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_find_dzl_sigmoid(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = al[i];
    dif = dif * (1.0 - dif);// sig(x) * ( 1 - sig(x) )
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_find_dzl_softmax(__global double *dwl_up, __global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = al[i];
    dif = dif * (1.0 - dif);// soft(x) * ( 1 - soft(x) )
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}