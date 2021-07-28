//
// Created by Henrique on 28-Jul-21.
//

#include "utils/time_utils.h"
#include <time.h>
double getns() {
	struct timespec t = {0};
	clock_gettime(CLOCK_MONOTONIC, &t);
	double ns = (double) t.tv_sec * 1.0e9 + t.tv_nsec;
	return ns;
}

double getms() {
	return getns() * 1.0e-6;
}
double getsec(){
	return getns()*1e-9;
}