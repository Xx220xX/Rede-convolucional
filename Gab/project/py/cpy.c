//
// Created by Henrique on 16/12/2021.
//

#include "cpy.h"

int PY_Cnn_new(Cnn *self) {
	*self = Cnn_new();
	return 0;
}



void PY_Cnn_out(Cnn self, float *p) {
	self->cm[self->l-1]->s->getvalues(self->cm[self->l-1]->s,p);
	self->cm[self->l-1]->s->print(self->cm[self->l-1]->s);
}
