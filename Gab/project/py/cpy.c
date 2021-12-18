//
// Created by Henrique on 16/12/2021.
//

#include "cpy.h"

int PY_Cnn_new(Cnn *self) {
	*self = Cnn_new();
	return 0;
}
