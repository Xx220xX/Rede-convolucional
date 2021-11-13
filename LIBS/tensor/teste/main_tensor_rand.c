#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include "tensor/tensor.h"
#include "png/png.h"



#include "gpu/Gpu.h"
int main() {
	srand(time(0));
	Gpu gpu	 = Gpu_new();
	int error = 0;
	cl_command_queue  queue = clCreateCommandQueueWithProperties(gpu->context,gpu->device,NULL,&error);
	Gpu_errormsg(error);
	Tensor t = Tensor_new(30, 30, 3, 1, TENSOR_INT,gpu->context,queue);

	t->randomize(t,TENSOR_NORMAL,2,0);
	char *tjson = t->json(t);
	FILE *f = fopen("../a.json","w");
	fprintf(f,"%s\n", tjson);
	free(tjson);
	fclose(f);
	int width = 150;
	int height = 150;
	char *image = calloc(width,height);
	memset(image,255,width*height);

	size_t bordas = 1;
	size_t pad = 5;

	size_t  w = (width - pad*(t->z-1)-2*bordas)/t->z  ;
	for (int z = 0; z < t->z; ++z) {
		t->imagegray(t,image,width,height,w,w,0,z*(w+pad)+bordas,z,0);
	}
	pngGRAY("../t.png",image,width,height);


	t->release(&t);
	gpu->release(&gpu);
	return 0;
}