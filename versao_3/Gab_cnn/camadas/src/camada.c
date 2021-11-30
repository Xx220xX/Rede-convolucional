//
// Created by hslhe on 13/11/2021.
//

#include "camadas/camada.h"

P3d internnal_getOutSize(Camada self) {
	return (P3d) {self->s->x, self->s->y, self->s->z};
}

void internal_Camada_new(Camada self, Gpu gpu, Queue queue, char layer_id, const char *layer_name, Parametros params,
						 Tensor entrada, P3d dim_in, P3d dim_out, Ecx erro) {
	erro->addstack(erro, "internal_Camada_new");
	self->erro = erro;
	self->a = entrada;
	self->size_in = dim_in;
	if (entrada) {
		self->da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, erro, 0, gpu->context, queue);
		if (self->erro->error)goto methods;
	}
	self->s = Tensor_new(dim_out.x, dim_out.y, dim_out.z, 1, erro, 0, gpu->context, queue);
	methods:
	memcpy((void *) &self->layer_id, &layer_id, sizeof(const char));
	memcpy(self, &layer_name, sizeof(const char *));
	self->maxcompute = &gpu->maxworks;
	self->params = params;
	erro->popstack(erro);
	self->getOutSize = (P3d (*)(void *)) internnal_getOutSize;

}

void internal_Camada_release(Camada *self) {
	if (!self)return;
	if (!*self)return;
	Release((*self)->da);
	Release((*self)->s);

}

char *internal_json(Camada self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = NULL;
	apendstr(string, len, "\"layer_name\":\"%s\",\n\"layer_id\":%d", self->layer_name, self->layer_id);
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
			 !self->params.skipLearn
	);

	return string;

}


void internal_saveCamada(FILE *f, Camada self) {
	fwrite(&self->layer_id, 1, sizeof(char), f);
	fwrite("#", 1, sizeof(char), f);
	uint32_t size_REAL = sizeof(REAL);
	fwrite(&size_REAL, 1, sizeof(uint32_t), f);
	fwrite(&self->params.hitlearn, 1, size_REAL, f);
	fwrite(&self->params.momento, 1, size_REAL, f);
	fwrite(&self->params.decaimento, 1, size_REAL, f);
	fwrite(&self->params.skipLearn, 1, sizeof(int), f);
	fwrite(&self->size_in, 1, sizeof(P3d), f);
}

#define REALCAST(dest, size_element, aux)\
    if((size_element) == sizeof(REAL))\
        (dest) = (aux).auxR;\
    else if((size_element) == sizeof(float)){\
        (dest) = (double)(aux).auxf;\
    }else\
        (dest) = (float)(aux).auxd

void internal_loadCamada(FILE *f, Parametros *parametros, P3d *size_in, uint32_t *size_element) {
	char flag;
	union {
		double auxd;
		float auxf;
		REAL auxR;
	} aux;

	fread(&flag, 1, sizeof(char), f);
	if (flag != '#') {
		fread(&flag, 1, sizeof(char), f);
	}
	fread(size_element, 1, sizeof(uint32_t), f);

	fread(&aux, 1, *size_element, f);
	REALCAST(parametros->hitlearn, *size_element, aux);
	fread(&aux, 1, *size_element, f);
	REALCAST(parametros->momento, *size_element, aux);
	fread(&aux, 1, *size_element, f);
	REALCAST(parametros->decaimento, *size_element, aux);
	fread(&parametros->skipLearn, 1, sizeof(int), f);
	fread(size_in, 1, sizeof(P3d), f);
}

void internal_saveTensor(FILE *f, Tensor t) {
	fwrite(&t->flag, sizeof(char), 1, f);
	fwrite(&t->length, sizeof(size_t), 1, f);
	fwrite(&t->bytes, sizeof(size_t), 1, f);
	void *data = t->getvalues(t, NULL);
	fwrite(data, 1, t->bytes, f);
	free_mem(data);
}

void internal_loadTensor(FILE *f, Tensor t, uint32_t size_element) {
	TensorFlag flag;
	size_t length;
	size_t bytes;

	fread(&flag.flag, sizeof(char), 1, f);
	fread(&length, sizeof(size_t), 1, f);
	fread(&bytes, sizeof(size_t), 1, f);
	void *data = alloc_mem(bytes, 1);
	REAL *dtaux;
	fread(data, 1, bytes, f);
	if (!(flag.inteiro || flag.caractere)) {
		if (size_element != sizeof(REAL)) {
			dtaux = alloc_mem(length, sizeof(REAL));
			if (size_element == sizeof(double)) {
				for (int i = 0; i < length; ++i)
					dtaux[i] = (REAL) ((double *) data)[i];
			} else {
				for (int i = 0; i < length; ++i)
					dtaux[i] = (REAL) ((float *) data)[i];
			}
			free_mem(data);
			data = dtaux;

		}
	}
	t->setvalues(t, data);
	free_mem(data);
}

void internal_saveREAL(FILE *f, REAL value) {
	fwrite(&value, sizeof(REAL), 1, f);
}

void internal_loadREAL(FILE *f, REAL *value, uint32_t size_element) {
	union {
		double auxd;
		float auxf;
		REAL auxR;
	} aux;
	fread(&aux, size_element, 1, f);
	if (size_element == sizeof(REAL))
		*value = aux.auxR;
	else if (size_element != sizeof(double)) {
		*value = (double) aux.auxf;
	} else
		*value = (float) aux.auxd;
}
