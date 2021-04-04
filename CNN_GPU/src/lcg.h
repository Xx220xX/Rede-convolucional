//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_LCG_H
#define GAB_LCG_H

typedef
unsigned long long int Bytes8;

typedef struct {
    Bytes8 a, c, m, rand_max, atual;
    int max_int;


} LCG;

LCG new_LCG(Bytes8 seed);

void LCG_setSeed(LCG *self, Bytes8 seed);

double LCG_randD(LCG *self);

int LCG_randI(LCG *self);

Bytes8 LCG_randB(LCG *self);


#endif //GAB_LCG_H
