
__kernel void reluativa(__global double *entrada,__global double *saida, int k0){
     int k = get_global_id(0) + k0;
     double v = entrada[k];
     if (v < 0)
        v = 0;
     saida[k] = v;
}

__kernel void relucalcgrad(__global double *gradentrada,__global double *entrada, __global double *gradnext, int k0){
     int k = get_global_id(0) + k0;
     gradentrada[k] = entrada[k] ? (0) : (1*gradnext[k]);
}


