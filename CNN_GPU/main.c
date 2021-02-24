//#define LOG_CNN_ADD_LAYERS
#include "src/cnn.h"
int main() {
    srand(time(0));
    Params p = {0.1, 0.99, 0.5};
    Cnn c = createCnnWithgpu("../kernels/gpu_functions_otmsize.cl", p, 16, 16, 3);
    CnnAddConvLayer(c, 2, 3, 8);
    CnnAddPoolLayer(c, 1, 3);
    CnnAddReluLayer(c);
    CnnAddFullConnectLayer(c, 10, FSIGMOIG);
    CnnAddDropOutLayer(c,0.5,time(0));
    // seleciona uma entrada e uma saida ambos vetores do tipo double
    //CnnCall(c, entrada);
    //CnnLearn(c,saidaCorreta);

    releaseCnn(&c);
    return 0;
}
