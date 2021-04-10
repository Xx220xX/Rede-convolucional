#define KV __kernel void
#define D __global double *

KV gab_feed_alan(D w, D a_donw, int m_a_donw, D b, D z, D a, int i0) {
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

KV gab_feed_tanh(D w, D a_donw, int m_a_donw, D b, D z, D a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = tanh(tmp);
}

KV gab_feed_relu(D w, D a_donw, int m_a_donw, D b, D z, D a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = tmp > 0.0 ? tmp : 0.0;
}

KV gab_feed_sigmoid(D w, D a_donw, int m_a_donw, D b, D z, D a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = 1.0 / (1 + exp(-tmp));
}

KV gab_feed_identify(D w, D a_donw, int m_a_donw, D b, D z, D a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = exp(tmp);
}

KV gab_feed_softmax(D w, D a_donw, int m_a_donw, D b, D z, D a, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
    a[i] = exp(tmp);
}

KV gab_feed(D w, D a_donw, int m_a_donw, D b, D z, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = b[i];
    for (int k = 0; k < m_a_donw; k++) {
        tmp = tmp + w[i * m_a_donw + k] * a_donw[k];
    }
    z[i] = tmp;
}

KV gab_normalize_alan(D a, D z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    if (tmp > 1)tmp = log(tmp - 1);
    else if (tmp < 1)tmp = -log(-tmp - 1);
    a[i] = tmp;
}

KV gab_normalize_tanh(D a, D z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = tanh(tmp);
}

KV gab_normalize_relu(D a, D z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = tmp > 0.0 ? tmp : 0.0;
}

KV gab_normalize_sigmoid(D a, D z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = 1.0 / (1.0 + exp(-tmp));
}

KV gab_normalize_identify(D a, D z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = tmp;
}

KV gab_normalize_softmax(D a, D z, double media, double desv, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = (z[i] - media) / desv;
    a[i] = exp(tmp);
}

KV gab_sum(D ans, D x, int n) {
    double tmp = 0;
    for (int i = 0; i < n; i++) {
        tmp += x[i];
    }
    ans[0] = tmp;
}

KV gab_desvio_padrao(D ans, D x, double media, int n) {
    double tmp = 0, aux;
    for (int i = 0; i < n; i++) {
        aux = (x[i] - media);
        tmp += aux * aux;
    }
    ans[0] = sqrt(tmp / (double) n);
}

KV gab_divide_vector(D ans, D x, double value, int i0) {
    int i = get_global_id(0) + i0;
    ans[i] = x[i] / value;
}

KV gab_find_last_dzl(D aL, D y, D dzL, D bL, double hit, int i0) {
    int i = get_global_id(0) + i0;
    dzL[i] = aL[i] - y[i];
    bL[i] = bL[i] - hit * dzL[i];
}

KV gab_find_dzl_alan(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, int i0) {
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

KV gab_find_dzl_tanh(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, int i0) {
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

KV gab_find_dzl_relu(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 0.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    if (zl[i] > 0)dif = 1;
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

KV gab_find_dzl_sigmoid(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, int i0) {
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

KV gab_find_dzl_identify(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, int i0) {
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

KV gab_find_dzl_softmax(D dwl_up, D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, int i0) {
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

KV gab_norm_find_dzl_alan(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, double media,double desv,double n,int i0) {
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

KV gab_norm_find_dzl_tanh(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit,double media,desv,int n,  int i0) {
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

KV gab_norm_find_dzl_relu(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, double media,double desv,int n,  int i0) {
    int i = get_global_id(0) + i0;
    double tmp = 0.0, dif = 0.0;
    for (int k = 0; k < m_dzl_up; ++k) {
        tmp += wl_up[k * m_dzl_up + i] * dzl_up[k];
        wl_up[k * m_dzl_up + i] = wl_up[k * m_dzl_up + i] - hit * dwl_up[k * m_dzl_up + i];
    }
    if (zl[i] > 0)dif = 1;
    double dif_desv = 1.0/(n*desv) * (zl[i] - media)*(1.0 - 2.0/n);
    dif = dif* ((1.0-1.0/n)*desv - (zl[i]-media)*dif_desv)/(desv*desv);
    tmp = tmp * dif;
    dzl[i] = tmp;
    bl[i] = bl[i] - hit * tmp;
}

KV gab_norm_find_dzl_identify(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, double media,double desv,int n,  int i0) {
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

KV gab_norm_find_dzl_sigmoid(D dwl_up,D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, double media,double desv,int n,  int i0) {
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

KV gab_norm_find_dzl_softmax(D dwl_up, D wl_up, D dzl_up, int m_dzl_up, D zl, D al, D dzl, D bl, double hit, double media,double desv, int n, int i0) {
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

KV gab_find_dwl(D dzl, D al_down, int ma_down, D dwl, int i0) {
    int i = get_global_id(0) + i0;
    int j = i % ma_down;
    i = i / ma_down;
    dwl[i * ma_down + j] = dzl[i] * al_down[j];
}

KV gab_find_dwl_and_update(D wl, D dzl, D al_down, int ma_down, D dwl, double hit, int i0) {
    int i = get_global_id(0) + i0;
    int j = i % ma_down;
    i = i / ma_down;
    dwl[i * ma_down + j] = dzl[i] * al_down[j];
    wl[i * ma_down + j] = wl[i * ma_down + j] - hit * dwl[i * ma_down + j];
}