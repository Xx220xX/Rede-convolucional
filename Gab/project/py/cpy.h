//
// Created by Henrique on 16/12/2021.
//

#ifndef GAB_CPY_H
#define GAB_CPY_H

#include "cnn/cnn_lua.h"

typedef struct {
	int epoca;
	int image;
	float progress;
	float erro;
	float a;
	float b;
} Info_callback;

int PY_Cnn_new(Cnn *self);

int PY_Cnn_release(Cnn self);

void PY_Cnn_out(Cnn self, float *p);

int PY_Cnn_lua(Cnn self, char *luaCommand);

int PY_Cnn_train(Cnn self, int epoca, int nbatch, float *input_values, float *target_values, int nsamples, Info_callback *info);

int PY_Cnn_force_end(Cnn self);

int PY_Cnn_predict(Cnn self, float *input_value, float *answer);

int PY_Cnn_seed(unsigned int long long seed);

int PY_Cnn_print(Cnn self);

#endif //GAB_CPY_H
