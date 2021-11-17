#include <png/png.h>
#include <math.h>
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
	Ecx ecx = Ecx_new(10);
	ecx->addstack(ecx,"clCreateCommandQueueWithProperties");
	cl_command_queue queue = clCreateCommandQueueWithProperties(gpu->context, gpu->device, NULL, &ecx->error);
	ecx->popstack(ecx);
	Tensor timage = Tensor_new(4, 5, 3, 1,ecx ,0, gpu->context,queue);

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
	ecx->addstack(ecx,"runRecursive");
	k->runRecursive(k,queue,timage->x*timage->y,gpu->maxworks,&timage->data,&timage->x,&timage->y);
	timage->erro->error = k->error;
	ecx->popstack(ecx);
	clFinish(queue);

	unsigned char * image = calloc(timage->x*timage->y,3);
	char *tjson = timage->json(timage,2);
	FILE *f = fopen("../a.json","w");
	fprintf(f,"%s\n", tjson);
	free(tjson);
	fclose(f);

	timage->imagegray(timage,image,timage->y,timage->x,timage->y,timage->x,0,0,0,0);
//	timage->imagegray(timage,image+(timage->x*timage->y),timage->y,timage->x,timage->y,timage->x,0,0,1,0);
//	timage->imagegray(timage,image+(2*timage->x*timage->y),timage->y,timage->x,timage->y,timage->x,0,0,2,0);
	pngRGB("../rgb.png",image,image+timage->x*timage->y,image+2*timage->x*timage->y,timage->x,timage->y);
	free(image);
	timage->release(&timage);
	gpu->release(&gpu);
	k->release(&k);
	ecx->print(ecx);
	ecx->release(&ecx);
	return 0;
}
