//
// Created by hslhe on 13/11/2021.
//

#include "camada.h"

Ponto3d internnal_getOutSize(Camada self) {
	return (Ponto3d) {self->s->x, self->s->y, self->s->z};
}

void internal_Camada_new(Camada self, Gpu gpu, Queue queue, int layer_id, const char *layer_name, Parametros params,
						 Tensor entrada, Ponto3d dim_in, Ponto3d dim_out, Ecx erro) {
	erro->addstack(erro, "internal_Camada_new");
	self->a = entrada;
	self->size_in = dim_in;
	if (entrada) {
		self->release_da = 1;
		self->da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, erro, 0, gpu->context, queue);
		if (self->erro->error)goto methods;
	}
	self->s = Tensor_new(dim_out.x, dim_out.y, dim_out.z, 1, erro, 0, gpu->context, queue);
	methods:
	memcpy((void *) &self->layer_id, &layer_id, sizeof(const int));
	memcpy(self, &layer_name, sizeof(const char *));
	self->maxcompute = &gpu->maxworks;
	self->erro = erro;
	self->params = params;
	erro->popstack(erro);
	self->getOutSize = (Ponto3d (*)(void *)) internnal_getOutSize;

}

void internal_Camada_release(Camada *self) {
	if (!self)return;
	if (!*self)return;
	if (self[0]->release_da) {
		(*self)->da->release(&(*self)->da);
	}
	if ((*self)->s)
		(*self)->s->release(&(*self)->s);

}

char *internal_json(Camada self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = NULL;
	apendstr(string, len, "\"layer_name\":%s,\n\"layer_id\":%d", self->layer_name, self->layer_id);
	if (self->a) {
		tmp = self->a->json(self->a, showValues);
		apendstr(string, len, ",\n"PAD"\"entrada\":%s", tmp);
		free_mem(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"entrada\":null");
	}
	if (self->da) {
		tmp = self->da->json(self->da, showValues);
		apendstr(string, len, ",\n"PAD"\"grad_entrada\":%s", tmp);
		free_mem(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"grad_entrada\":null");
	}
	if (self->s) {
		tmp = self->s->json(self->s, showValues);
		apendstr(string, len, ",\n"PAD"\"saida\":%s", tmp);
		free_mem(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"saida\":null");
	}
	apendstr(string, len, ",\n"PAD"\"max_compute\":%zu,\n"
			PAD"\"params\":{\"hitlearn\":%g,\"momento\":%g,\"decaimento\":%g,\"treinavel\":%d}",
			 *self->maxcompute, (double) self->params.hitlearn, (double) self->params.momento, (double) self->params.decaimento,
			 !self->params.disable_learn
	)

	return string;

}
