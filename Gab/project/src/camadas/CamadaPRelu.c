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
	if (!self_p) {
		return;
	}
	if (!*self_p) {
		return;
	}
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->W);
	Release((*self_p)->dW);
	Release((*self_p)->preluativa);
	Release((*self_p)->preluonlyfix);
	Release((*self_p)->prelucalcgrad);
	Release((*self_p)->prelucalcgradBatch);
	Release((*self_p)->preluonlyDABatch);
	Release((*self_p)->kernel_fixW);
	gab_free(*self_p);
	*self_p = NULL;
}

static int CamadaPRelu_propagation(CamadaPRelu self) {
	Execute(preluativa, self->super.s->length, &self->super.a->data, &self->super.s->data, &self->W->data);
	return self->super.ecx->error;
}

static int CamadaPRelu_backpropagation(CamadaPRelu self, Tensor ds) {
	if (self->super.da) {
		int learn = !self->super.params.skipLearn;
		Execute(prelucalcgrad, self->super.da->length, &self->super.da->data, &self->super.a->data, &ds->data, &self->W->data, &self->dW->data, &learn, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	} else if (!self->super.params.skipLearn) {
		Execute(preluonlyfix, self->dW->length, &self->super.a->data, &ds->data, &self->W->data, &self->dW->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	}

	return self->super.ecx->error;
}

static int CamadaPRelu_backpropagationBatch(CamadaPRelu self, Tensor ds, size_t batchSize) {
	if (self->super.da) {
		//Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize
		Execute(prelucalcgradBatch, self->super.da->length, &self->super.da->data, &self->super.a->data, &ds->data, &self->W->data, &self->dW->data, &batchSize);
	} else if (!self->super.params.skipLearn) {
		//Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize
		Execute(preluonlyDABatch, self->dW->length, &self->super.a->data, &ds->data, &self->W->data, &self->dW->data, &batchSize);
	}

	return self->super.ecx->error;
}

static int CamadaPRelu_learnBatch(CamadaPRelu self) {
	if (!self->super.params.skipLearn) {
		//kernel_fixW(Vector w, Vector dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0)
		Execute(kernel_fixW, self->dW->length, &self->W->data, &self->dW->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	}
	return self->super.ecx->error;
}

static char *CamadaPRelu_json(CamadaPRelu self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = NULL;
	apendstr(string, len, "{\n");
	tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, PAD"%s", tmp);
	gab_free(tmp);
	apendTensor("A", W, string, len, tmp, showValues);
	apendTensor("dA", dW, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaPRelu_getGenerate(CamadaPRelu self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s(Params(%g,%g,%g,%d),RDP(%d,%g,%g))", lname, (double) self->super.params.hitlearn, (double) self->super.params.momento, (double) self->super.params.decaimento, self->super.params.skipLearn, self->rdp_a.type, (double) self->rdp_a.a, (double) self->rdp_a.b);
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
	if (self->super.ecx->error) {
		goto end;
	}
	self->super.ecx->addstack(self->super.ecx, "CamadaPRelu_save");
	internal_saveCamada(f, (Camada) self);
	internal_saveTensor(f, self->W);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaPRelu_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaPRelu_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	CamadaPRelu self = (CamadaPRelu) CamadaPRelu_new(gpu, queue, size_in, entrada, parametros, RDP(-1), ecx);
	internal_loadTensor(f, self->W, size_element);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}
int CamadaPRelu_fprintf(CamadaPRelu self, FILE * destino, char *format, ...){
	va_list  v;
	va_start(v,format);
	internal_Camada_fprint(self,destino,format,v);
	fprintf(destino,"W -> ");self->W->fprint(self->W,destino);
	fprintf(destino,"dW -> ");self->dW->fprint(self->dW,destino);
	return 0;
}
Camada CamadaPRelu_new(Gpu gpu, Queue queue, P3d size_in, Tensor entrada, Parametros params, RandomParams rdp_a, Ecx ecx) {
	ECXPUSH(ecx);
	CamadaPRelu self = gab_alloc(1, sizeof(CamadaPRelu_t));
	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, PRELU_ID, lname, params, entrada, size_in, size_out, ecx);
	self->rdp_a = rdp_a;
	self->W = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->dW = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	if (rdp_a.type != -1) {
		if (rdp_a.type == 0) {
			rdp_a = internal_getDefaultRDP(1, size_in.x * size_in.y * size_in.z, self->super.s->length);
		}
		self->super.ecx->error = self->W->randomize(self->W, rdp_a.type, rdp_a.a, rdp_a.b);
		if (ecx->error) {
			goto methods;
		}
	}

	if (ecx->error) {
		goto methods;
	}
	//kV preluativa(Vector entrada, Vector saida, Vector A, int k0)
	KRN_news(self->preluativa, "preluativa", "Vector entrada, Vector saida, Vector A, int k0");
	//prelucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0)
	KRN_news(self->prelucalcgrad, "prelucalcgrad", "Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0");
	//preluonlyfix(Vector entrada, Vector gradnext, Vector A, Vector dA, REAL hitlearn, REAL momento, REAL decaimento, int k0)
	KRN_news(self->preluonlyfix, "preluonlyfix", "Vector entrada, Vector gradnext, Vector A, Vector dA, REAL hitlearn, REAL momento, REAL decaimento, int k0");
	//prelucalcgradBatch(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize, int k0)
	KRN_news(self->prelucalcgradBatch, "prelucalcgradBatch", "Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize, int k0");
	//preluonlyDABatch(Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize, int k0)
	KRN_news(self->preluonlyDABatch, "preluonlyDABatch", "Vector entrada, Vector gradnext, Vector A, Vector dA, long batchSize, int k0");
	//kernel_fixW(Vector w, Vector dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0)
	KRN_news(self->kernel_fixW, "kernel_fixW", "Vector w, Vector dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0");


	ECXPOP(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaPRelu_release;
	self->super.propagation = (int (*)(void *)) CamadaPRelu_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaPRelu_backpropagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaPRelu_backpropagationBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) CamadaPRelu_learnBatch;
	self->super.json = (char *(*)(void *, int)) CamadaPRelu_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaPRelu_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaPRelu_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaPRelu_fprintf;
	return (Camada) self;
}
