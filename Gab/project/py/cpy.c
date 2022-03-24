//
// Created by Henrique on 16/12/2021.
//

#include <math.h>
#include "cpy.h"

int PY_Cnn_new(Cnn *self) {
//	printf("c cnn new\n");
	*self = Cnn_new();
	return 0;
}

int PY_Cnn_release(Cnn self) {
//	printf("c cnn release\n");
	return self->release(&self);
}

void PY_Cnn_out(Cnn self, float *p) {
	self->cm[self->l - 1]->s->getvalues(self->cm[self->l - 1]->s, p);
//	self->cm[self->l-1]->s->print(self->cm[self->l-1]->s);
}

int PY_Cnn_lua(Cnn self, char *luaCommand) {
	int error = CnnLuaLoadString(self, luaCommand);
	return error;
}

int PY_Cnn_train(Cnn self, int epoca, int nbatch, float *input_values, float *target_values, int nsamples, Info_callback *info) {
	self->running = 1;
	int ep_atual;
	int im_atual;
	int bt;
	int bt_max = nsamples / nbatch;
	int i;
	self->setMode(self, 1);
	P3d s_in = self->size_in, s_ou = self->getSizeOut(self);
	Info_callback local = {0};
	if (info) {
		local = *info;
	}
	local.b = 1-local.a;
	float  erro;
	double mse = 1;
	double a =local.a;
	double b = local.b;
	Tensor inp = Tensor_new(s_in.x, s_in.y, s_in.z, 1, self->ecx, 0, self->gpu->context, self->queue);
	Tensor out = Tensor_new(s_ou.x, s_ou.y, s_ou.z, 1, self->ecx, 0, self->gpu->context, self->queue);
	for (ep_atual = 0; self->running && !self->ecx->error && ep_atual < epoca; ++ep_atual) {
		im_atual = 0;
		for (bt = 0; self->running && !self->ecx->error && bt < bt_max; ++bt) {
			for (i = 0; self->running && !self->ecx->error && i < nbatch; ++i) {
				im_atual = bt * nbatch + i;
				out->setvalues(out, target_values + im_atual * out->length);
				self->predictv(self, input_values + im_atual * inp->length);
				self->learnBatch(self, out, nbatch);
				erro = self->mse(self);
				if (info) {
					mse = (mse * a) + (erro *b);
					local.erro = mse;
					local.epoca = ep_atual+1;
					local.image = im_atual+1;
					local.progress = (ep_atual * nsamples + im_atual+1) / ((float) epoca * nsamples) * 100;
					*info = local;
				}else{
					printf("\r%.4f",erro);
				}
				if(isnan(erro)){
					self->running = 0;
					self->jsonF(self,1,"c1.json");
					printf("\n");
					printf("nbatch %d\n",nbatch);
					printf("im_atual %d\n",im_atual);
					printf("i %d\n",i);
					printf("ep_atual %d\n",ep_atual);
					printf("bt %d\n",bt);
					printf("out->length %zu\n",out->length);
					printf("inp->length %zu\n",inp->length);
				}
			}
			self->fixBatch(self);

		}
		bt = nsamples - im_atual -1 ;
		if(bt>0) {
			for (; self->running && !self->ecx->error && im_atual < nsamples; im_atual++) {
				out->setvalues(out, target_values + im_atual * out->length);
				self->predictv(self, input_values + im_atual * inp->length);
				self->learnBatch(self, out, nbatch);
				erro = self->mse(self);
				if (info) {
					mse = (mse * a) + (erro *b);
					local.erro = mse;
					local.epoca = ep_atual+1;
					local.image = im_atual+1;
					local.progress = (ep_atual * nsamples + im_atual+1) / ((float) epoca * nsamples) * 100;
					*info = local;
				}else{
					printf("\r%.4f",erro);
				}

			}
			self->fixBatch(self);
		}
	}

	self->setMode(self, 0);
	self->running = 0;
	Release(inp);
	Release(out);
	return self->ecx->error;
}

int PY_Cnn_force_end(Cnn self) {
	self->running = 0;
	return 0;
}


int PY_Cnn_predict(Cnn self, float *input_value, float *answer) {
	self->setMode(self, 0);
	self->predictv(self, input_value);
	self->cm[self->l - 1]->s->getvalues(self->cm[self->l - 1]->s, answer);
	return self->ecx->error;
}

int PY_Cnn_seed(unsigned long long int seed) {
	LCG_setSeed(seed);
	return 0;
}

int PY_Cnn_print(Cnn self) {
	self->print(self, NULL);
	return 0;
}
