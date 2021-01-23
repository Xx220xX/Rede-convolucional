/***
    Created by Xx220xX on 04/05/2020.
    This file is to opencl gpu amd
*/

//#pragma OPENCL EXTENSION cl_amd_fp64:enable

#define FUNC_ID_TANH 0
#define FUNC_ID_DFTANH 1
#define FUNC_ID_RELU 2
#define FUNC_ID_DFRELU 3
#define FUNC_ID_SIGMOID 8
#define FUNC_ID_DFSIGMOID 9
#define FUNC_ID_ALAN 16
#define FUNC_ID_DFALAN 17
#define FLAG_DIF 1


typedef double *MatGPU;

double alan(double x);

double dfalan(double x);

double dftanh(double x);

double sigmoid(double x);

double dfsigmoid(double x);

double relu(double x);

double dfrelu(double x);


typedef double (*double_func_double)(double);


double_func_double getfunction_by_id(int id) {
    switch (id) {
        case FUNC_ID_ALAN:
            return alan;
            break;
        case FUNC_ID_DFALAN:
            return dfalan;
            break;
        case FUNC_ID_TANH :
            return tanh;
            break;
        case FUNC_ID_DFTANH :
            return dftanh;
            break;
        case FUNC_ID_RELU:
            return relu;
            break;
        case FUNC_ID_DFRELU:
            return dfrelu;
            break;
        case FUNC_ID_SIGMOID:
            return sigmoid;
            break;
        case FUNC_ID_DFSIGMOID:
            return dfsigmoid;
            break;
        default:
            return NULL;
    }

}

__kernel void iter_call(__global double *al, int ma, int na,
                        __global double *zl, int mz, int nz,
                        __global double *wl, int mwl, int nw,
                        __global double *a_down, int ma_down, int na_down,
                        __global double *bl, int mb, int nb,
                        int id_func_activation) {

    double_func_double ativate = getfunction_by_id(id_func_activation);
    int i, j, k;
    for (i = 0; i < mwl; i++) {
        for (j = 0; j < na_down; j++) {
            zl[i * na_down + j] = bl[i * na_down + j];
            for (k = 0; k < ma_down; k++) {
                zl[i * na_down + j] += wl[i * ma_down + k] * a_down[k * na_down + j];
            }
            al[i * na_down + j] = ativate(zl[i * na_down + j]);
        }
    }
}

// aprende a saida
__kernel void last_layer_learn(__global double *dzL, int mdz,
                               __global double *aL,
                               __global double *out,
                               __global double *dwL,
                               __global double *a_L_1, int ma_L_1,
                               __global double *w,
                               __global double *b) {
    int i, j;
    for (i = 0; i < mdz; i++) {
        // dzL = AL - OUT
        dzL[i] = aL[i] - out[i];
        // dwL = dzl * aL-1^T
        for (j = 0; j < ma_L_1; j++) {
            dwL[i * ma_L_1 + j] = dzL[i] * a_L_1[j];
        }
    }

}

// aprende a camada interna
__kernel void iter_aprende(__global double *dzl, int m_dzl,
                           __global double *wl_up, int n_wl_up,
                           __global double *dzl_up, int m_dzl_up, int n_dzl_up,
                           __global double *zl, int n_dzl,
                           __global double *al_down, int m_al_down,
                           __global double *dwl,
                           __global double *wl,
                           __global double *bl,
                           int id_ativate_function) {
    double_func_double dif = getfunction_by_id(id_ativate_function | FLAG_DIF);
    // dz[l] = (w[l+1]^t * dz[l+1] ) X f'(z[l])
    int i, j, k;
    for (i = 0; i < n_wl_up; i++) {
        dzl[i] = 0.0;
        for (k = 0; k < m_dzl_up; k++) {
            dzl[i] += wl_up[k * n_wl_up + i] * dzl_up[k * n_dzl_up];
        }
        dzl[i] = dzl[i] * dif(zl[i]);
    }
    // dw[l] = dz[l]*a[l-1]^t

    for (i = 0; i < m_dzl; i++) {
        for (j = 0; j < m_al_down; j++) {
            dwl[i * m_al_down + j] = dzl[i] * al_down[j];
        }
    }

}

__kernel void ajusta_pesos(__global double *dzl, int size_b,
                           __global double *dwl, int size_w,
                           __global double *w,
                           __global double *b,
                           double hit_learn) {
    int i;
    for (i = 0; i < size_b; ++i) {
        b[i] = b[i] - hit_learn * dzl[i];
    }
    for (i = 0; i < size_w; ++i) {
        w[i] = w[i] - hit_learn * dwl[i];
    }
}


double alan(double x) {
    if (x < -2.0) return -log(-x + 0.6);
    if (x > 2.0)
        return
                log(x + 0.6);
    return
            tanh(x);
}

double dfalan(double x) {
    if (x < -2.0)
        return
                1.0 / (-x + 0.6);
    if (x > 2.0)
        return
                1.0 / (x + 0.6);
    double ch = cosh(x);
    return
            1.0 / (ch * ch);
}

double dftanh(double x) {
    double ch = cosh(x);
    return
            1.0 / (ch * ch);
}

double sigmoid(double x) {
    x = 1 + exp(-x);
    return
            1.0 / (1 + exp(-x));
}

double dfsigmoid(double x) {
    x = exp(-x);
    return
            x / ((1.0 + x) * (1.0 + x));
}

double relu(double x) {
    if (x > 0)
        return
                x;
    return
            0.0;
}

double dfrelu(double x) {
    if (x > 0)
        return
                1;
    return
            0.0;
}
