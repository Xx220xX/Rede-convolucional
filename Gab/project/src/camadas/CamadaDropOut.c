//
// Created by hslhe on 19/11/2021.
//

/**
 * Implementa a camda dropOut
 * Essa camada aplica uma probabilidade p de que a entrada apareça na saída.
 */
#include "camadas/CamadaDropOut.h"

static const char *lname = "DropOut";

static void CamadaDropOut_release(CamadaDropOut *self_p) {
	if (!self_p) {
		return;
	}
	if (!*self_p) {
		return;
	}
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->hitmap);
	gab_free(*self_p);
	*self_p = NULL;
}


static Tensor CamadaDropOut_propagation_predict(CamadaDropOut self, Tensor a) {
	return a;
}

static Tensor CamadaDropOut_propagation(CamadaDropOut self, Tensor a) {
	Execute(dropativa, self->super.s->length, &a->data, &self->super.s->data, &self->hitmap->data, &self->seed, &self->probabilidade_saida);
	self->seed += self->super.s->length;
	self->seed = (self->seed * 0x5deece66dULL + 0xbULL) & ((1ULL << 31) - 1);
	return self->super.s;
}

static int CamadaDropOut_backpropagation(CamadaDropOut self, Tensor ds) {
	if (self->super.da) {
		Execute(dropcalcgrad, self->super.da->length, &self->super.da->data, &self->hitmap->data, &ds->data);
	}

	return self->super.ecx->error;
}

static char *CamadaDropOut_json(CamadaDropOut self, int showValues) {
	char *string = NULL;
	char *tmp = internal_json((Camada) self, showValues);
	int len = 0;
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"seed\":%llu,\n"
			PAD"\"probabilidade_saida\":%g", tmp, self->seed, (double) self->probabilidade_saida);


	gab_free(tmp);

	apendTensor("hitmap", hitmap, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaDropOut_getGenerate(CamadaDropOut self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s(%g,%llu)", lname, (double) self->probabilidade_saida, self->seed);

	return string;
}

static int CamadaDropOut_save(CamadaDropOut self, FILE *f) {
	if (self->super.ecx->error) {
		goto end;
	}
	self->super.ecx->addstack(self->super.ecx, "CamadaDropOut_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->seed, sizeof(cl_long), 1, f);
	internal_saveREAL(f, self->probabilidade_saida);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaDropOut_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaDropOut_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;
	cl_long seed;
	REAL probabilidade_saida;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&seed, sizeof(cl_long), 1, f);
	internal_loadREAL(f, &probabilidade_saida, size_element);

	CamadaDropOut self = (CamadaDropOut) CamadaDropOut_new(gpu, queue, size_in, probabilidade_saida, seed, entrada, ecx);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}

int CamadaDropOut_fprintf(CamadaDropOut self, FILE *destino, char *format, ...) {
	va_list v;
	va_start(v, format);
	internal_Camada_fprint(self, destino, format, v);
	fprintf(destino, "hitmap -> ");
	self->hitmap->fprint(self->hitmap, destino);
	return 0;
}


void CamadaDropout_setMode(CamadaDropOut self, int istraing) {
	if (istraing) {
		self->super.propagation = (Tensor (*)(void *, Tensor)) CamadaDropOut_propagation_predict;
	} else {
		self->super.propagation = (Tensor (*)(void *, Tensor)) CamadaDropOut_propagation;
	}
}

Camada CamadaDropOut_new(Gpu gpu, Queue queue, P3d size_in, REAL probabilidade_saida, cl_ulong seed, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaDropOut_new");
	CamadaDropOut self = gab_alloc(1, sizeof(CamadaDropOut_t));
	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, DROPOUT_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);
	self->seed = seed;
	self->probabilidade_saida = probabilidade_saida;
	self->hitmap = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, TENSOR_CHAR, gpu->context, queue);

	if (ecx->error) {
		goto methods;
	}

	KRN_news(self->dropativa, "dropativa", "Vector entrada, Vector saida, __global char *hitmap, long seed, REAL pativa, int k0");
	KRN_news(self->dropcalcgrad, "dropcalcgrad", "Vector gradentrada, __global char *hitmap, Vector gradnext, int k0");

	ECXPOP(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaDropOut_release;
	self->super.propagation = (Tensor (*)(void *, Tensor)) CamadaDropOut_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaDropOut_backpropagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) internal_notBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) internal_unused;
	self->super.json = (char *(*)(void *, int)) CamadaDropOut_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaDropOut_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaDropOut_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaDropOut_fprintf;
	self->setMode  = CamadaDropout_setMode;
	return (Camada) self;
}
