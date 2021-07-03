//
// Created by Henrique on 18-Jun-21.
//

#include "../src/LCG_Random/lcg.h"
#include <time.h>
#include <stdio.h>

int main(){
	LCG_setSeed(time(NULL));
	int v[50] ;
	for(int i=0;i<50;v[i++] = (int)i-1);
	LCG_shuffle(v,50,sizeof (int ));
	for(int i=0;i<50;printf("%d ",v[i++]));
	printf("\n");
	return 0;
}