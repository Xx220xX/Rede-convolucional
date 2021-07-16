#define __kernel
#define __global
#define get_global_id(x) global_index_
int global_index_ = 0;

#define RUN_KERNEL(kernel,iteractions,v0,...) \
	for(global_index_ = 0;global_index_ <iteractions;global_index_++) \
                             kernel(v0, ## __VA_ARGS__)
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "camadas/utils.h"
#include "camadas/pool.h"
#include "camadas/conv.h"
#include "camadas/fullconnect.h"
int main(){
	double gradIn[8];
	double entrada[8] = {0,2,3,5,4,8,9,9};
	double saida[3] = {2,5,9};
	int py=3;
	int fy=3;
	int iny = 8;
	int sy = 3;
	RUN_KERNEL(poolCalcGrads,8,entrada,gradIn,saida,saida,1,fy,1,py,1,iny,1,sy,0);
	for(int i=0;i<8;i++){
		printf("%lf ",gradIn[i]);
	}
	return 0;
}