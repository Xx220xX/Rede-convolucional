#include <stdio.h>
#include "cnn.h"
#include "camadas/camada.h"

int main() {
	Gpu gpu = Gpu_new();
	Ecx ecx = Ecx_new(10);
	Queue  queue = gpu->Queue_new(gpu,&ecx->error);

	clReleaseCommandQueue(queue);
	ecx->release(&ecx);
	gpu->release(&gpu);
	return 0;
}