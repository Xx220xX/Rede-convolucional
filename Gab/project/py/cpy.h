//
// Created by Henrique on 16/12/2021.
//

#ifndef GAB_CPY_H
#define GAB_CPY_H
#include "cnn/cnn_lua.h"
int PY_Cnn_new(Cnn *self);
void PY_Cnn_out(Cnn self,float *p);
#endif //GAB_CPY_H