//
// Created by hslhe on 14/11/2021.
//

#include <stdio.h>
#include <string.h>
#include "tensor/exc.h"
#include "gpu/Gpu.h"
#include "gpu/Kernel.h"


int main() {
	Ecx ecx = Ecx_new(18);
	Gpu gpu = Gpu_new();
	gpu->compileProgram(gpu,
						UTILS_MACRO_KERNEL
						"KV test(Vector a,int c, char d,long s, __global unsigned char * v){int x = 10;}");
	Kernel kn = Kernel_news(gpu->program, "test",
							"Vector notas,int c, char d,long s, __global unsigned char * v");

	kn->release(&kn);
	gpu->release(&gpu);
	ecx->release(&ecx);
	return 0;
}