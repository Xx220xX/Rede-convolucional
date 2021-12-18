//
// Created by hslhe on 13/11/2021.
//

#ifndef GAB_CNN_PARAMETROS_H
#define GAB_CNN_PARAMETROS_H

#include "config.h"

typedef struct Parametos_t {
	REAL hitlearn, momento, decaimento;
	int skipLearn;
} Parametros;

/// Params(hitlearn,momento = 0.0,decaimento = 0.0, skipLearn = 0)
#define Params(hitlearn, ...)((Parametros){hitlearn,## __VA_ARGS__})
#endif //GAB_CNN_PARAMETROS_H
