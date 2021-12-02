//
// Created by Henrique on 24/11/2021.
//

#include "cnn/kernel_lib.h"
#include "kernels/defaultkernel.h"
const char *KERNEL_LIB_get_defaultKernel() {
	return __default_kernel__;
}

