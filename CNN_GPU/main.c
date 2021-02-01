#include <stdio.h>
#include "src/cnn.h"

int main() {

    Params p = {0.1,0.99,0.5};
    WrapperCL cl ;
    WrapperCL_initbyFile(&cl,"../kernels/gpu_functions.cl");
    WrapperCL_release(&cl);
//    Cnn c = createCnn
//            (&cl,p,28,28,3);
//    CnnAddConvLayer(c,1,3,8);
//
//    releaseCnn(&c);

    return 0;
}
