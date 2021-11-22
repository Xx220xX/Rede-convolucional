//
// Created by henrique on 19/11/2021.
//

/***
 * Implementa uma versão da relu em que possui valores treináveis,
 * onde cada dimensão possui sua inclinação
 */
#include "camadas/CamadaPRelu.h"

static const char *lname = "PRelu";

static void CamadaPRelu_release(CamadaPRelu *self_p) {
	if (!self_p)return;
	if (!*self_p)return;
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->A);
	Release((*self_p)->dA);
	free_mem(*self_p);
	*self_p = NULL;
}

static int CamadaPRelu_propagation(CamadaPRelu self) {
	Execute(preluativa, self->super.s->length,
			&self->super.a->data, &self->super.s->data,
			&self->A
	);
	return self->super.erro->error;
}

static int CamadaPRelu_backpropagation(CamadaPRelu self, Tensor ds) {
	if (self->super.da) {
		int learn = !self->super.params.skipLearn;
		Execute(prelucalcgrad, self->super.da->length,
				&self->super.da->data, &self->super.a->data,
				&ds->data, &self->A->data, &self->dA->data,
				&learn,
				&self->super.params.hitlearn,
				&self->super.params.momento,
				&self->super.params.decaimento
		);
	} else if (!self->super.params.skipLearn) {
		Execute(preluonlyfix, self->dA->length,
				&self->super.a->data, &ds->data,
				&self->A->data, &self->dA->data,
				&self->super.params.hitlearn,
				&self->super.params.momento,
				&self->super.params.decaimento
		);
	}

	return self->super.erro->error;
}

static char *CamadaPRelu_json(CamadaPRelu self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = NULL;
	apendstr(string, len, "{\n");
	tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, ",\n"PAD"%s", tmp);
	free_mem(tmp);
	apendTensor("A", A, string, len, tmp, showValues);
	apendTensor("dA", dA, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaPRelu_getGenerate(CamadaPRelu self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s(Params(%g,%g,%g,%d),RDP(%d,%g,%g))",
			 lname,
			 (double) self->super.params.hitlearn,
			 (double) self->super.params.momento,
			 (double) self->super.params.decaimento,
			 self->super.params.skipLearn,
			 self->rdp_a.type,
			 (double) self->rdp_a.a,
			 (double) self->rdp_a.b
	);
	return string;
}

/**
 * Salva a camada conv em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 bytes -> separação deve ser '#'
 * 4 bytes -> size_element
 * 8 bytes -> length
 * size_element*length bytes -> data
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
static int CamadaPRelu_save(CamadaPRelu self, FILE *f) {
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaPRelu_save");
	fwrite(&self->super.layer_id, 1, sizeof(int), f);
	fwrite("#", 1, 1, f);
	fwrite(&self->A->size_element, 1, sizeof(uint32_t), f);
	fwrite(&self->A->length, 1, sizeof(size_t), f);

	void *data = self->A->getvalues(self->A, NULL);
	fwrite(data, self->A->size_element, self->A->length, f);
	free_mem(data);
	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

Camada CamadaPRelu_new(Gpu gpu, Queue queue, P3d size_in, Tensor entrada, Parametros params, RdP rdp_a, Ecx ecx) {
	ecx->addstack(ecx, "CamadaPRelu_new");
	CamadaPRelu self = alloc_mem(1, sizeof(CamadaPRelu_t));
	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, PRELU_ID, lname, params, entrada, size_in, size_out, ecx);
	self->rdp_a = rdp_a;
	self->A = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->dA = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	if (rdp_a.type != -1) {
		if (rdp_a.type == 0) {
			rdp_a.type = TENSOR_NORMAL;
			rdp_a.a = (REAL) (2.0) / (REAL) (size_in.x * size_in.y * size_in.z);
			rdp_a.b = 0;
		}
		self->super.erro->error = self->A->randomize(self->A, rdp_a.type, rdp_a.a, rdp_a.b);
		if (ecx->error)goto methods;
	}

	if (ecx->error)goto methods;
	self->preluativa = Kernel_news(gpu->program, "preluativa",
								   "Vector entrada, Vector saida, Vector A, int k0");
	CheckKernel(preluativa);

	self->prelucalcgrad = Kernel_news(gpu->program, "prelucalcgrad",
									  "Vector gradentrada, Vector entrada, Vector gradnext,"
									  "Vector A, Vector dA,\n"
									  "int learn, REAL hitlearn, REAL momento,\n"
									  "REAL decaimento,\n"
									  "int k0");
	CheckKernel(prelucalcgrad);

	self->preluonlyfix = Kernel_news(gpu->program, "preluonlyfix",
									 "Vector entrada, Vector gradnext, Vector A, Vector dA,\n"
									 "REAL hitlearn, REAL momento,\n"
									 "REAL decaimento,\n"
									 "int k0");

	CheckKernel(preluonlyfix);

	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaPRelu_release;
	self->super.propagation = (int (*)(void *)) CamadaPRelu_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaPRelu_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaPRelu_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaPRelu_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaPRelu_save;
	return (Camada) self;
}
