#ifndef CNN_GPU_UTILS_H
#define CNN_GPU_UTILS_H
#define FOR3D(i,j,k,iF,jF,kF) \
    for(int i=0;i<iF;i++)\
    for(int j=0;j<jF;j++)\
    for(int k=0;k<kF;k++)

#define FOR2D(i,j,iF,jF) \
    for(int i=0;i<iF;i++)\
    for(int j=0;j<jF;j++)



#define callocdouble(x)(double*)calloc(x,sizeof(double))

// random de -1 a 1
#define RANDOM_BILATERAL() (2.0*(rand() / ((double) RAND_MAX))-1)
#endif