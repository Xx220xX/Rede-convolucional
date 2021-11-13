#include <png/png.h>
#include "tensor/tensor.h"
#include "gpu/Gpu.h"
#include "gpu/Kernel.h"

#define get_global_id(x) 0

KV im(Vector image, int ix, int iy, int k0) {
	int k = get_global_id(0) + k0;
	int x,y;
	KTensorRemap2D(k,x,y,iy)
	image[KTensorMap(x,y,0,ix,iy)] =  255.0 * ((REAL)!(x & y));
	image[KTensorMap(x,y,1,ix,iy)] = (REAL)(x^y);
	image[KTensorMap(x,y,2,ix,iy)] = (REAL)(x|y);
}




int main() {
	Gpu gpu = Gpu_new();
	int error = 0;
	cl_command_queue queue = clCreateCommandQueueWithProperties(gpu->context, gpu->device, NULL, &error);
	if (error)exit(error);
	Tensor timage = Tensor_new(255, 255, 3, 1, 0, gpu->context,queue);

//	for (int i = 0; i < timage->x*timage->y; ++i) {
//		im(timage->data,timage->x,timage->y,i);
//	}
	char *prog = UTILS_MACRO_KERNEL
				 "KV im(Vector image, int ix, int iy, int k0) {\n"
				 "\tint k = get_global_id(0) + k0;\n"
				 "\tint x,y;\n"
				 "\tKTensorRemap2D(k,x,y,iy)\n"
				 "\timage[KTensorMap(x,y,0,ix,iy)] =  255.0 * ((REAL)!(x & y));\n"
				 "\timage[KTensorMap(x,y,1,ix,iy)] = (REAL)(x^y);\n"
				 "\timage[KTensorMap(x,y,2,ix,iy)] = (REAL)(x|y);\n"
				 "}";
	gpu->compileProgram(gpu,prog);
	Kernel k = Kernel_new(gpu->program,"im",4,KP,KI,KI,KI,KI);
	k->runRecursive(k,queue,timage->x*timage->y,gpu->maxworks,&timage->data,&timage->x,&timage->y);
	clFinish(queue);
	unsigned char * image = calloc(timage->x*timage->y,3);
	char *tjson = timage->json(timage);
	FILE *f = fopen("../a.json","w");
	fprintf(f,"%s\n", tjson);
	free(tjson);
	fclose(f);

	timage->imagegray(timage,image,timage->x,timage->y,timage->x,timage->y,0,0,0,0);
	timage->imagegray(timage,image+(timage->x*timage->y),timage->x,timage->y,timage->x,timage->y,0,0,1,0);
	timage->imagegray(timage,image+(2*timage->x*timage->y),timage->x,timage->y,timage->x,timage->y,0,0,2,0);
	pngRGB("../rgb.png",image,image+timage->x*timage->y,image+2*timage->x*timage->y,timage->x,timage->y);
	free(image);
	timage->release(&timage);
	gpu->release(&gpu);
	k->release(&k);
	return 0;
}
