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

	gab_free(*self_p);
	*self_p = NULL;
}

static Tensor CamadaRelu_propagation(CamadaRelu self,Tensor a) {
	self->super.a = a;
	Execute(reluativa, self->super.s->length,
			&self->super.a->data, &self->super.s->data,
			&self->lessoh, &self->greateroh
	);
	return self->super.s;
}

static int CamadaRelu_backpropagation(CamadaRelu self, Tensor ds) {
	if (self->super.da) {
		Execute(relucalcgrad, self->super.da->length,
				&self->super.da->data, &self->super.a->data, &ds->data,
				&self->lessoh, &self->greateroh
		);
	}
	return self->super.ecx->error;
}

static char *CamadaRelu_json(CamadaRelu self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{"
					 PAD"%s,\n"
					 PAD"\"lessoh\":%g,\n"
					 PAD"\"greateroh\":%g",
			 tmp,
			 (double) self->lessoh, (double) self->greateroh);

	gab_free(tmp);
	apendstr(string, len, "\n}");
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
	if (self->super.ecx->error)goto end;
	self->super.ecx->addstack(self->super.ecx, "CamadaRelu_save");
	internal_saveCamada(f, (Camada) self);
	internal_saveREAL(f,self->lessoh);
	internal_saveREAL(f,self->greateroh);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaRelu_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaRelu_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;

	REAL lessoh,greateroh;

	internal_loadCamada(f, &parametros, &size_in, &size_element);
	internal_loadREAL(f,&lessoh,size_element);
	internal_loadREAL(f,&greateroh,size_element);

	CamadaRelu self = (CamadaRelu) CamadaRelu_new(gpu, queue, size_in,lessoh,greateroh ,entrada, ecx);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}
int CamadaRelu_fprintf(CamadaRelu self, FILE * destino, char *format, ...){
	va_list  v;
	va_start(v,format);
	internal_Camada_fprint(self,destino,format,v);
	fprintf(destino,"%f %f\n",self->lessoh,self->greateroh);
	return 0;
}
Camada CamadaRelu_new(Gpu gpu, Queue queue, P3d size_in, REAL less, REAL greater, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaRelu_new");
	CamadaRelu self = gab_alloc(1, sizeof(CamadaRelu_t));
	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, RELU_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);
	self->lessoh = less;
	self->greateroh = greater;
	KRN_news(self->reluativa , "reluativa",
			 "Vector entrada, Vector saida, REAL menor, REAL maior, int k0");

	KRN_news(self->relucalcgrad , "relucalcgrad",
									 "Vector gradentrada, Vector entrada, Vector gradnext,"
									 " REAL menor, REAL maior, int k0");



	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaRelu_release;
	self->super.propagation = (Tensor (*)(void *, Tensor)) CamadaRelu_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaRelu_backpropagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) internal_notBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) internal_unused;
	self->super.json = (char *(*)(void *, int)) CamadaRelu_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaRelu_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaRelu_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaRelu_fprintf;
	return (Camada) self;
}

