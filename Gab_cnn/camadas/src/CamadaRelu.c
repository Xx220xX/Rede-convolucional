//
// Created by hslhe on 19/11/2021.
//
/***
 * Implemente a função relu
 * Essa camada possilita o uso de inclinações diferentes para a função
 * se x < 0
 *  y  =  x*lessoh
 * caso contrario
 * 	y = x*greateroh
 */

#include "camadas/CamadaRelu.h"

static const char *lname = "Relu";

static void CamadaRelu_release(CamadaRelu *self_p) {
	if (!self_p)return;
	if (!*self_p)return;
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->reluativa);
	Release((*self_p)->relucalcgrad);

	free_mem(*self_p);
	*self_p = NULL;
}

static int CamadaRelu_propagation(CamadaRelu self) {
	Execute(reluativa, self->super.s->length,
			&self->super.a->data, &self->super.s->data,
			&self->lessoh, &self->greateroh
	);
	return self->super.erro->error;
}

static int CamadaRelu_backpropagation(CamadaRelu self, Tensor ds) {
	if (self->super.da) {
		Execute(relucalcgrad, self->super.da->length,
				&self->super.da->data, &self->super.a->data, &ds->data,
				&self->lessoh, &self->greateroh
		);
	}
	return self->super.erro->error;
}

static char *CamadaRelu_json(CamadaRelu self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = NULL;
	apendstr(string, len,
			 "{"
					 PAD"\"lessoh\":%g,\n"
					 PAD"\"greateroh\":%g,\n",
			 (double) self->lessoh, (double) self->greateroh);

	tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, ",\n"PAD"%s\n}", tmp);
	free_mem(tmp);
	return string;
}

static char *CamadaRelu_getGenerate(CamadaRelu self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s(%g,%g)",
			 lname,
			 (double) self->lessoh, (double) self->greateroh
	);

	return string;
}

/**
 * Salva a camada relu em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 bytes -> separação deve ser '#'
 * 8 bytes -> lessoh
 * 8 bytes -> greateroh
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
static int CamadaRelu_save(CamadaRelu self, FILE *f) {
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaRelu_save");
	fwrite(&self->super.layer_id, 1, sizeof(int), f);
	fwrite("#", 1, 1, f);
	double p = (double)self->lessoh;
	fwrite(&p, 1, sizeof(double ), f);
	p = (double)self->greateroh;
	fwrite(&p, 1, sizeof(double ), f);
	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

Camada CamadaRelu_new(Gpu gpu, Queue queue, P3d size_in, float less, float greater, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaRelu_new");
	CamadaRelu self = alloc_mem(1, sizeof(CamadaRelu_t));
	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, RELU_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);
	self->lessoh = less;
	self->greateroh = greater;
	self->reluativa = Kernel_news(gpu->program, "reluativa",
								  "Vector entrada, Vector saida, REAL menor, REAL maior, int k0");
	CheckKernel(reluativa);

	self->relucalcgrad = Kernel_news(gpu->program, "relucalcgrad",
									 "Vector gradentrada, Vector entrada, Vector gradnext,"
									 " REAL menor, REAL maior, int k0");

	CheckKernel(relucalcgrad);


	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaRelu_release;
	self->super.propagation = (int (*)(void *)) CamadaRelu_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaRelu_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaRelu_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaRelu_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaRelu_save;
	return (Camada) self;
}

