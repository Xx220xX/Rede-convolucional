
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
