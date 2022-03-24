//
// Created by hslhe on 18/11/2021.
//

#include "camadas/CamadaPool.h"

static const char *lname = "Pooling";

void CamadaPooling_release(CamadaPool *selfp) {
	internal_Camada_release((Camada *) (selfp));
	Release((*selfp)->poolCalcGrads);
	Release((*selfp)->poolativa);
	Release((*selfp)->hitmap);
	gab_free(*selfp);
}

Tensor CamadaPooling_propagation(CamadaPool self, Tensor a) {
	self->super.a = a;
	Execute(poolativa, self->super.s->length, &self->super.a->data, &self->super.s->data, &self->hitmap->data, &self->passox, &self->passoy, &self->filtrox, &self->filtroy, &self->super.s->x, &self->super.s->y, &self->super.a->x, &self->super.a->y);
    ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
    return self->super.s;
}

int CamadaPooling_backpropagation(CamadaPool self, Tensor ds) {
	if (self->super.da) {
//		Super.da->fill(Super.da,0);
		Execute(poolCalcGrads, self->super.da->length,
				&self->super.a->data, &self->super.da->data, &ds->data, &self->super.s->data, &self->filtrox, &self->filtroy, &self->passox, &self->passoy, &self->super.a->x, &self->super.a->y, &self->super.s->x, &self->super.s->y
			   );
	}
	return self->super.ecx->error;
}

int CamadaPooling_backpropagationmnx(CamadaPool self, Tensor ds) {
	if (self->super.da) {
//		Super.da->fill(Super.da,0);
		Execute(poolCalcGrads, self->super.da->length,
				&self->super.a->data, &self->super.da->data, &ds->data, &self->hitmap->data, &self->filtrox, &self->filtroy, &self->passox, &self->passoy, &self->super.a->x, &self->super.a->y, &self->super.s->x, &self->super.s->y
			   );
	}
    ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return self->super.ecx->error;
}

char *CamadaPooling_json(CamadaPool self, int showValues) {
    ECX_RETURN_IF_ERROR(Super.ecx, NULL)
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"passo\":[%zu,%zu],\n"
			PAD"\"filtro\":[%zu,%zu],\n"
			PAD"\"type\":\"%s\""
			   "\n}", tmp, self->passox, self->passoy, self->filtrox, self->filtroy, self->type == MAXPOOL ? "Max Poling" : (self->type == MINPOOL ? "Min Poling" : "Average Pooling"));
	gab_free(tmp);
    ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return string;
}

char *CamadaPooling_getGenerate(CamadaPool self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s(P2D(%zu,%zu),P2D(%zu,%zu),%d)", lname, self->passox, self->passoy, self->filtrox, self->filtroy, self->type);
	return string;
}

/**
 * Salva a camada pool em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 byte -> separação deve ser '#'
 * 8 bytes -> passo x
 * 8 bytes -> passo y
 * 8 bytes -> filtro x
 * 8 bytes -> filtro y
 *
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
int CamadaPooling_save(CamadaPool self, FILE *f) {
    ECX_RETURN_IF_ERROR(Super.ecx, Super.ecx->error)
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->type, 1, sizeof(uint32_t), f);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->filtrox, 1, sizeof(size_t), f);
	fwrite(&self->filtroy, 1, sizeof(size_t), f);

    ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return self->super.ecx->error;
}

Camada CamadaPool_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
    ECX_RETURN_IF_ERROR(ecx, NULL)
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;

	uint32_t type;
	P2d passo;
	P2d filtro;

	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&type, 1, sizeof(uint32_t), f);
	fread(&passo.x, 1, sizeof(size_t), f);
	fread(&passo.y, 1, sizeof(size_t), f);
	fread(&filtro.x, 1, sizeof(size_t), f);
	fread(&filtro.y, 1, sizeof(size_t), f);
	CamadaPool self = (CamadaPool) CamadaPool_new(gpu, queue, passo, filtro, size_in, type, entrada, ecx);
	end:
    ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return (Camada) self;
}

int CamadaPool_fprintf(CamadaPool self, FILE *destino, char *format, ...) {
	va_list v;
	va_start(v, format);
	internal_Camada_fprint(self, destino, format, v);
	return 0;
}

Camada CamadaPool_new(Gpu gpu, Queue queue, P2d passo, P2d filtro, P3d size_in, uint32_t type_pooling, Tensor entrada, Ecx ecx) {
    ECX_RETURN_IF_ERROR(ecx, NULL)
	CamadaPool self = gab_alloc(1, sizeof(CamadaPool_t));

	P3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1, size_in.z};
	internal_Camada_new((Camada) self, gpu, queue, POOL_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);
	self->hitmap = Tensor_new(Super.s->x, Super.s->y, Super.s->z, Super.s->w, Super.ecx, 0, gpu->context, queue);
	self->passox = passo.x;
	self->passoy = passo.y;
	self->type = type_pooling;
	self->filtrox = filtro.x;
	self->filtroy = filtro.y;

	if (type_pooling == MAXPOOL) {
		KRN_news(self->poolativa, "poolativa", "Vector entrada, Vector saida, Vector hmap,\n"
											   "int passox,int passoy,\n"
											   "int filtrox,int filtroy,\n"
											   "int saidatx, int saidaty,\n"
											   "int entradatx, int entradaty, int k0");


		KRN_news(self->poolCalcGrads, "poolCalcGrads", "Vector entrada, Vector gradEntrada,\n"
													   "Vector gradNext, Vector hmap,\n"
													   "int fx, int fy, int px, int py,\n"
													   "int entradatx, int entradaty,\n"
													   "int saidatx, int saidaty,\n"
													   "int k0");
		self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaPooling_backpropagationmnx;


	} else if (type_pooling == AVEPOOL) {
		KRN_news(self->poolativa, "poolAVativa", "Vector entrada, Vector saida,\n"
												 "int passox,int passoy,\n"
												 "int filtrox,int filtroy,\n"
												 "int saidatx, int saidaty,\n"
												 "int entradatx, int entradaty, int k0");

		KRN_news(self->poolCalcGrads, "poolAvCalcGrads", "Vector entrada, Vector gradEntrada,\n"
														 "Vector gradNext, Vector saida,\n"
														 "int fx, int fy, int px, int py,\n"
														 "int entradatx, int entradaty,\n"
														 "int saidatx, int saidaty,\n"
														 "int k0");
		self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaPooling_backpropagation;

	} else if (type_pooling == MINPOOL) {
		KRN_news(self->poolativa, "poolativaMin", "Vector entrada, Vector saida,Vector hmap,\n"
												  "int passox,int passoy,\n"
												  "int filtrox,int filtroy,\n"
												  "int saidatx, int saidaty,\n"
												  "int entradatx, int entradaty, int k0");


		KRN_news(self->poolCalcGrads, "poolCalcGrads", "Vector entrada, Vector gradEntrada,\n"
													   "Vector gradNext, Vector hmap,\n"
													   "int fx, int fy, int px, int py,\n"
													   "int entradatx, int entradaty,\n"
													   "int saidatx, int saidaty,\n"
													   "int k0");
		self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaPooling_backpropagationmnx;


	} else {
		ecx->setError(ecx, GAB_INVALID_PARAM, "Tipo invalido\n");
		goto methods;
	}
	methods:
    ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	self->super.release = (void (*)(void *)) CamadaPooling_release;
	self->super.propagation = (Tensor (*)(void *, Tensor)) CamadaPooling_propagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) internal_notBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) internal_unused;
	self->super.json = (char *(*)(void *, int)) CamadaPooling_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaPooling_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaPooling_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaPool_fprintf;
	return (Camada) self;
}
