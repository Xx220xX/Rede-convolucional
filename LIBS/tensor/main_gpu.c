//
// Created by hslhe on 13/11/2021.
//

#include <gpu/Gpu.h>

#include "tensor/tensor.h"
int main(){
	Gpu gpu = Gpu_new();

	Ecx ecx = Ecx_new(5);
	cl_command_queue  queue = clCreateCommandQueueWithProperties(gpu->context,gpu->device,NULL,&ecx->error);
	Tensor t = Tensor_new(3,3,1,1,ecx,TENSOR_RAM,gpu->context,queue);

	REAL f = 3.141592;
	t->fillM(t,0,t->bytes,&f,sizeof(REAL));
	char *json = t->json(t,2);
	printf("%s\n",json);
	free_mem(json);
	ecx->print(ecx);
	t->release(&t);
	gpu->release(&gpu);
	ecx->release(&ecx);
	return 0;
}
