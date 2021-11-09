//
// Created by hslhe on 07/11/2021.
//
#define __global
#define __kernel
int __X__ = 0;
#define get_global_id(i)__X__

#include <stdio.h>
#include <float.h>
#include <math.h>


#include"camadas/utils.h"
#include"camadas/bathnorm.h"
#include"camadas/cnnutils.h"
#include"camadas/conv.h"
#include"camadas/convf.h"
#include"camadas/convNc.h"
#include"camadas/dropout.h"
#include"camadas/fullconnect.h"
#include"camadas/padding.h"
#include"camadas/pool.h"
#include"camadas/poolav.h"
#include"camadas/prelu.h"
#include"camadas/relu.h"
#include"camadas/softmax.h"

void printv(REAL *v,int len){
	for(int  i=0;i<len;i++)
		printf("%.2lf ",(double)v[i]);
	printf("\n");
}
int main() {
	REAL entrada[] = {0.83, 0.38, -0.30,0.99, -0.80, 0.44, 0.71, 0.48, 0.53};
	REAL filtro[] = {-0.03, 0.11,0.06, 0.08,0.05, -0.14,-0.04, 0.18};
	REAL z[2*2*2] ={0};
	REAL s[2*2*2] = {0};
	int ex = 3,ey = 3;
	int sx = 2,sy = 2;
	int fx = 2,fy = 2,fz = 2;
	int px = 1,py = 1;
	for (int i=0;i<8;i++) {
		__X__ = i;
		convFSum(filtro, entrada, z, s, px, py, sx, sy, ex, ey, fx, fy, fz, 2, 0);
	}
	printv(z,8);
	printv(s,8);


	return 0;
}