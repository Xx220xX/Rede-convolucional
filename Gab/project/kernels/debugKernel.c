//
// Created by Henrique on 20/11/2021.
//
#define __kernel
#define __global
int globI = 0;
#include <math.h>
#include <stdio.h>
#include "float.h"


int get_global_id(int id) {
	globI++;
	return globI - 1;
}

#include "kernels/camadas/utils.h"
#include "kernels/camadas/bathnorm.h"
#include "kernels/camadas/cnnutils.h"
#include "kernels/camadas/conv.h"
#include "kernels/camadas/convf.h"
#include "kernels/camadas/convNc.h"
#include "kernels/camadas/dropout.h"
#include "kernels/camadas/fullconnect.h"
#include "kernels/camadas/padding.h"
#include "kernels/camadas/poolav.h"
#include "kernels/camadas/poolMax.h"
#include "kernels/camadas/poolMin.h"
#include "kernels/camadas/prelu.h"
#include "kernels/camadas/relu.h"
#include "kernels/camadas/softmax.h"

int main() {
	return 0;
}