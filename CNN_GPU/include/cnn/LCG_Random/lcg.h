//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_LCG_H
#define GAB_LCG_H

#include <stdlib.h>
#include "utils/memory_utils.h"

#define LCG_NORMAL 2
#define LCG_UNIFORM 1
typedef unsigned long long int Bytes64;

typedef struct {
	Bytes64 a, c, m, rand_max, atual;
	int max_int;
} LCG;

LCG new_LCG(Bytes64 seed);

void pLCG_setSeed(LCG *self, Bytes64 seed);

double pLCG_randD(LCG *self);

int pLCG_randI(LCG *self);

Bytes64 pLCG_randB(LCG *self);

void LCG_setSeed(Bytes64 seed);

double LCG_randD();

double pLCG_randn(LCG *self);

double LCG_randn();

int LCG_randI();

Bytes64 LCG_randB();

void LCG_shuffle(void *d, size_t n, size_t size_element);

void pLCG_shuffle(LCG *self, void *d, size_t n, size_t size_element);

#endif //GAB_LCG_H
