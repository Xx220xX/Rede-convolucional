//
// Created by hslhe on 13/11/2021.
//

#include <gpu/Gpu.h>


int main(){
	Gpu gpu = Gpu_new();
	CLInfo  info = gpu->getClInfo(gpu);
	char *infojs = info.json(&info);
	printf("%s\n",infojs);
	free_mem(infojs);










	gpu->release(&gpu);
	return 0;
}
