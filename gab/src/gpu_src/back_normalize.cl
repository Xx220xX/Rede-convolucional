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