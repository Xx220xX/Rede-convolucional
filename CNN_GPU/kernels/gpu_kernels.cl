__kernel void gab_feed_alan(__global double *w, __global double *a_donw, int m_a_donw, __global double *b, __global double *z, __global double *a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    if (tmp > 1.0)tmp = log(tmp - 1);
    else if (tmp < 1) tmp = -log(-tmp - 1);
    a[i] = tmp;
}

__kernel void gab_feed_tanh(__global double *w, __global double *a_donw, int m_a_donw, __global double *b, __global double *z, __global double *a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = tanh(tmp);
}

__kernel void gab_feed_relu(__global double *w, __global double *a_donw, int m_a_donw, __global double *b, __global double *z, __global double *a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = tmp > 0.0 ? tmp : 0.0;
}

__kernel void gab_feed_sigmoid(__global double *w, __global double *a_donw, int m_a_donw, __global double *b, __global double *z, __global double *a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = 1.0 / (1 + exp(-x));
}

__kernel void gab_feed_identify(__global double *w, __global double *a_donw, int m_a_donw, __global double *b, __global double *z, __global double *a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = exp(tmp);
}

__kernel void gab_feed_softmax(__global double *w, __global double *a_donw, int m_a_donw, __global double *b, __global double *z, __global double *a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = exp(tmp);
}

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

__kernel void gab_sum(__global double *ans, __global double *x, int n) {
    double tmp = 0;
    for (int i = 0; i < n; i++) {
        tmp += x[i];
    }
    ans[0] = tmp;
}

__kernel void gab_desvio_padrao(__global double *ans, __global double *x, double media, int n) {
    double tmp = 0, aux;
    for (int i = 0; i < n; i++) {
        aux = (x[i] - media);
        tmp += aux * aux;
    }
    ans[0] = sqrt(tmp / (double) n);
}

__kernel void gab_divide_vector(__global double *ans, __global double *x, double value, int i0) {
    int i = get_global_id(0) + i0;
    ans[i] = x[i] / value;
}

__kernel void gab_find_last_dzl(__global double *aL, __global double *y, __global double *dzL, __global double *bL, double hit, int i0) {
    int i = get_global_id(0) + i0;
    dzL[i] = aL[i] - y[i];
    bL[i] = bL[i] - hit * dzL[i];
}

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

__kernel void gab_find_dzl_identify(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
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

__kernel void gab_norm_find_dzl_alan(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, double media,double desv,double n,int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = (zl[i] - media)/desv;
    if (dif > 1.0)dif = 1.0 / (dif - 1);
    else if (dif < 1.0)dif = 1.0 / (-dif - 1);
    double dif_desv = 1.0/(n*desv) * (zl[i] - media)*(1.0 - 2.0/n);
    dif = dif* ((1.0-1.0/n)*desv - (zl[i]-media)*dif_desv)/(desv*desv);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_norm_find_dzl_tanh(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit,double desv,  int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = al[i];
    dif = 1 - dif * dif;// 1 - tanh(x)^2
    double dif_desv = 1.0/(n*desv) * (zl[i] - media)*(1.0 - 2.0/n);
    dif = dif* ((1.0-1.0/n)*desv - (zl[i]-media)*dif_desv)/(desv*desv);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_norm_find_dzl_relu(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, double media,double desv,  int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 0.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    if (z[i] > 0)dif = 1;
    double dif_desv = 1.0/(n*desv) * (zl[i] - media)*(1.0 - 2.0/n);
    dif = dif* ((1.0-1.0/n)*desv - (zl[i]-media)*dif_desv)/(desv*desv);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_norm_find_dzl_identify(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, double media,double desv,  int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    double dif_desv = 1.0/(n*desv) * (zl[i] - media)*(1.0 - 2.0/n);
    dif = dif* ((1.0-1.0/n)*desv - (zl[i]-media)*dif_desv)/(desv*desv);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_norm_find_dzl_sigmoid(__global double *dwl_up,__global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, double media,double desv,  int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = al[i];
    dif = dif * (1.0 - dif);// sig(x) * ( 1 - sig(x) )
    double dif_desv = 1.0/(n*desv) * (zl[i] - media)*(1.0 - 2.0/n);
    dif = dif* ((1.0-1.0/n)*desv - (zl[i]-media)*dif_desv)/(desv*desv);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

__kernel void gab_norm_find_dzl_softmax(__global double *dwl_up, __global double *wl_up, __global double *dzl_up, int m_dzl_up, __global double *zl, __global double *al, __global double *dzl, __global double *bl, double hit, double media,double desv,  int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 1.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    dif = al[i];
    dif = dif * (1.0 - dif);// soft(x) * ( 1 - soft(x) )
    double dif_desv = 1.0/(n*desv) * (zl[i] - media)*(1.0 - 2.0/n);
    dif = dif* ((1.0-1.0/n)*desv - (zl[i]-media)*dif_desv)/(desv*desv);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
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