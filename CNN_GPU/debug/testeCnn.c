//
// Created by petel on 22/05/2021.
//

#include "../include/cnn/cnn.h"

int main(){
	Params p = {0.1, 0.0, 0.0, 1};
	Cnn c = createCnnWithWrapper("../kernels/gpu_function.cl",p,5,5,2,CL_DEVICE_TYPE_ALL);

	releaseCnn(&c);
}