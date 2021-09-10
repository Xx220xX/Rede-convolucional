//
// Created by Xx220xX on 12/05/2020.
//

#include "LCG_Random/lcg.h"
#include <string.h>
#include <math.h>

#define MAX_LCG_RAND ((1ULL << 48)-1)
LCG default_var_lcg = {0x5DEECE66DULL, 11ULL, (1ULL << 48), (1ULL << 48) - 1, 1 << 5, 1 << (sizeof(int) * 8 - 1)};

LCG new_LCG(Bytes64 seed) {
	LCG self = {0};
	self.a = 0x5DEECE66DULL;
	self.c = 11ULL;
	self.m = (1ULL << 48);
	self.rand_max = self.m - 1;
	self.atual = seed;
	self.max_int = 1 << (sizeof(int) * 8 - 1);
	return self;
}

void pLCG_setSeed(LCG *self, Bytes64 seed) {
	*self = new_LCG(seed);
	pLCG_randB(self);
}

Bytes64 pLCG_randB(LCG *self) {
	self->atual = (self->a * self->atual + self->c) % self->m;
	return self->atual;
}

int pLCG_randI(LCG *self) {
	return (int) (LCG_randB(self) % self->max_int);
}

double pLCG_randD(LCG *self) {
	return (double) LCG_randB(self) / self->rand_max;
}


void LCG_setSeed(Bytes64 seed) {
	pLCG_setSeed(&default_var_lcg, seed);
}

double LCG_randD() {
	return pLCG_randD(&default_var_lcg);
}

int LCG_randI() {
	return pLCG_randI(&default_var_lcg);
}

Bytes64 LCG_randB() {
	return pLCG_randB(&default_var_lcg);
}

void pLCG_shuffle(LCG *self, void *d, size_t n, size_t size_element) {
	char *tmp = alloc_mem(1, size_element);
	char *arr = d;
	if (n > 1) {
		size_t i, rnd, j;
		for (i = 0; i < n - 1; ++i) {
			rnd = ceil(pLCG_randD(self) * (n - i - 1));
			j = i + rnd;
			memcpy(tmp, arr + j * size_element, size_element);
			memcpy(arr + j * size_element, arr + i * size_element, size_element);
			memcpy(arr + i * size_element, tmp, size_element);
		}
	}
	free_mem(tmp);
}

void LCG_shuffle(void *d, size_t n, size_t size_element) {
	pLCG_shuffle(&default_var_lcg, d, n, size_element);
}

double pLCG_randn(LCG *self) {
	//https://en.wikipedia.org/wiki/Box–Muller_transform
	double u1 = pLCG_randD(self);
	double u2 = pLCG_randD(self);
	double Z0 = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
	return Z0;
}

double LCG_randn() {
	return pLCG_randn(&default_var_lcg);
}
