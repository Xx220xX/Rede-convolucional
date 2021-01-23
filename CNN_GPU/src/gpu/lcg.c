//
// Created by Xx220xX on 12/05/2020.
//

#include "lcg.h"

LCG new_LCG(Bytes8 seed) {
    LCG self = {0};
    self.a = 0x5DEECE66DULL;
    self.c = 11ULL;
    self.m = (1ULL << 48);
    self.rand_max = self.m - 1;
    self.max_int =1<<(sizeof(int)*8-1);
    self.atual = seed;
    return self;
}

void LCG_setSeed(LCG *self,Bytes8 seed) {
    self->atual = seed;
}

Bytes8 LCG_randB(LCG * self) {
    self->atual = (self->a * self->atual + self->c) % self->m;
    return self->atual;
}

int LCG_randI(LCG *self) {
    return (int) LCG_randB(self) % self->max_int;
}

double LCG_randD(LCG *self) {
    return (double) LCG_randB(self) / self->rand_max;
}
