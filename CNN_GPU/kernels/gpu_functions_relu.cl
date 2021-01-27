
__kernel void reluativa(__global double *entrada,__global double *saida,
                        int entradatx, int entradaty, int entradatz, int saidatx, int saidaty){
     for (int i = 0; i < entradatx; i++)
            for (int j = 0; j < entradaty; j++)
                for (int z = 0; z <entradatz; z++) {
                    double v = entrada[TensorAT(i, j, z, entradatx, entradaty)];
                    if (v < 0)
                        v = 0;
                    saida[TensorMap(i, j, z, saidatx, saidaty)] = v;
                }
}

__kernel void relucalcgrad(__global double *gradentrada,__global double *entrada, __global double *gradnext,
                           int entradatx, int entradaty, int entradatz){
     for (int i = 0; i < entradatx; i++)
            for (int j = 0; j < entradaty; j++)
                for (int z = 0; z < entradatz; z++) {
                   gradentrada[TensorAT(i, j, z)] = (entrada[TensorAT(i,j,z)]) ?  (0) : (1*gradnext[TensorAT(i, j, z)]);
                }
}


