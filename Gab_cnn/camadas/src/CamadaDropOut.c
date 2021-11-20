//
// Created by hslhe on 19/11/2021.
//

/**
 * Implementa a camda dropOut
 * Essa camada aplica uma probabilidade p de que a entrada apareça na saída.
 */
#include "camadas/CamadaDropOut.h"

static const char *lname = "CamadaDropOut";

static void CamadaDropOut_release(CamadaDropOut *self_p) {
	if (!self_p)return;
	if (!*self_p)return;
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->hitmap);
	free_mem(*self_p);
	*self_p = NULL;
}

static int CamadaDropOut_propagation(CamadaDropOut self) {
	Execute(dropativa, self->super.s->lenght,
			&self->super.a->data, &self->hitmap->data, &self->seed,
			&self->probabilidade_saida
	);
	self->seed += self->super.s->lenght;
	self->seed = (self->seed * 0x5deece66dULL + 0xbULL) & ((1ULL << 31) - 1);
	return self->super.erro->error;
}

static int CamadaDropOut_backpropagation(CamadaDropOut self, Tensor ds) {
	if (self->super.da) {
		Execute(dropcalcgrad, self->super.da->lenght,
				&self->super.da->data,
				&self->hitmap->data,
				&ds->data
		);
	}

	return self->super.erro->error;
}

static char *CamadaDropOut_json(CamadaDropOut self, int showValues) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "{"
					 PAD"\"seed\":%llu,\n"
					 PAD"\"probabilidade_saida\":%g,\n",
			 self->seed,
			 (double) self->probabilidade_saida);

	char *tmp = NULL;
	apendTensor("hitmap", hitmap, string, len, tmp, showValues);

	tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, ",\n"PAD"%s\n}", tmp);
	free_mem(tmp);
	return string;
}

static char *CamadaDropOut_getGenerate(CamadaDropOut self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s(%g,%llu)",
			 lname,
			 (double) self->probabilidade_saida, self->seed
	)

	return string;
}

/**
 * Salva a camada conv em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 bytes -> separação deve ser '#'
 * 8 bytes -> seed
 * 8 bytes -> probabilidade de saída (double)
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
static int CamadaDropOut_save(CamadaDropOut self, FILE *f) {
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaDropOut_save");
	fwrite(&self->super.layer_id, 1, sizeof(int), f);
	fwrite("#", 1, 1, f);
	fwrite(&self->seed, 1, sizeof(cl_ulong), f);
	double p = (double) self->probabilidade_saida;
	fwrite(&p, 1, sizeof(double), f);
	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

Camada CamadaDropOut_new(Gpu gpu, Queue queue, P3d size_in,
						 REAL probabilidade_saida, cl_ulong seed,
						 Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaDropOut_new");
	CamadaDropOut self = alloc_mem(1, sizeof(CamadaDropOut_t));
	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, DROPOUT_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);
	self->seed = seed;
	self->probabilidade_saida = probabilidade_saida;
	self->hitmap = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, TENSOR_CHAR, gpu->context, queue);

	if (ecx->error)goto methods;

	self->dropativa = Kernel_news(gpu->program, "dropativa",
								  "Vector entrada, Vector saida, __global char *hitmap, long seed,\n"
								  "REAL pativa, int k0");
	CheckKernel(dropativa);

	self->dropcalcgrad = Kernel_news(gpu->program, "dropcalcgrad",
									 "Vector gradentrada, __global char *hitmap, Vector gradnext, int k0");

	CheckKernel(dropcalcgrad);

	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaDropOut_release;
	self->super.propagation = (int (*)(void *)) CamadaDropOut_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaDropOut_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaDropOut_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaDropOut_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaDropOut_save;
	return (Camada) self;
}
