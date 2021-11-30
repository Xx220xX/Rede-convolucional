//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_LCG_H
#define GAB_LCG_H

#include <stdlib.h>
#include "config.h"

#define LCG_NORMAL 2
#define LCG_UNIFORM 1
typedef unsigned long long int Bytes64;

typedef struct {
	Bytes64 a, c, m, rand_max, atual;
	int max_int;
} LCG;

extern LCG new_LCG(Bytes64 seed);

extern void pLCG_setSeed(LCG *self, Bytes64 seed);

extern REAL pLCG_randD(LCG *self);

extern int pLCG_randI(LCG *self);

extern Bytes64 pLCG_randB(LCG *self);

extern void LCG_setSeed(Bytes64 seed);

extern REAL LCG_randD();

extern REAL pLCG_randn(LCG *self);

extern REAL LCG_randn();

extern int LCG_randI();

extern Bytes64 LCG_randB();

extern void LCG_shuffle(void *d, size_t n, size_t size_element);

extern void pLCG_shuffle(LCG *self, void *d, size_t n, size_t size_element);

#endif //GAB_LCG_H
