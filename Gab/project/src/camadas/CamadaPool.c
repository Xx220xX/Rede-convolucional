//
// Created by hslhe on 18/11/2021.
//

#include "camadas/CamadaPool.h"

static const char *lname = "Pooling";

void CamadaPooling_release(CamadaPool *selfp) {
	internal_Camada_release((Camada *) (selfp));
	Release((*selfp)->poolCalcGrads);
	Release((*selfp)->poolativa);
	gab_free(*selfp);
}

int CamadaPooling_propagation(CamadaPool self) {
	Execute(poolativa, self->super.s->length,

			&self->super.a->data, &self->super.s->data, &self->passox, &self->passoy, &self->filtrox, &self->filtroy, &self->super.s->x, &self->super.s->y, &self->super.a->x, &self->super.a->y

		   );
	return self->super.ecx->error;
}

int CamadaPooling_backpropagation(CamadaPool self, Tensor ds) {
	if (self->super.da) {
		Execute(poolCalcGrads, self->super.da->length,

				&self->super.a->data, &self->super.da->data, &ds->data, &self->super.s->data, &self->filtrox, &self->filtroy, &self->passox, &self->passoy, &self->super.a->x, &self->super.a->y, &self->super.s->x, &self->super.s->y

			   );
	}
	return self->super.ecx->error;
}

char *CamadaPooling_json(CamadaPool self, int showValues) {
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
	if (self->super.ecx->error) { goto end; }
	self->super.ecx->addstack(self->super.ecx, "CamadaPooling_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->type, 1, sizeof(uint32_t), f);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->filtrox, 1, sizeof(size_t), f);
	fwrite(&self->filtroy, 1, sizeof(size_t), f);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaPool_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaPooling_save");
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
	ecx->popstack(ecx);
	return (Camada) self;
}

Camada CamadaPool_new(Gpu gpu, Queue queue, P2d passo, P2d filtro, P3d size_in, uint32_t type_pooling, Tensor entrada, Ecx ecx) {
	ECXPUSH(ecx);
	CamadaPool self = gab_alloc(1, sizeof(CamadaPool_t));

	P3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1, size_in.z};
	internal_Camada_new((Camada) self, gpu, queue, POOL_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);
	self->passox = passo.x;
	self->passoy = passo.y;
	self->type = type_pooling;
	self->filtrox = filtro.x;
	self->filtroy = filtro.y;

	if (type_pooling == MAXPOOL) {
		KRN_new(self->poolativa, "poolativa", "Vector entrada, Vector saida,\n"
											  "int passox,int passoy,\n"
											  "int filtrox,int filtroy,\n"
											  "int saidatx, int saidaty,\n"
											  "int entradatx, int entradaty, int k0");


		KRN_new(self->poolCalcGrads, "poolCalcGrads", "Vector entrada, Vector gradEntrada,\n"
													  "Vector gradNext, Vector saida,\n"
													  "int fx, int fy, int px, int py,\n"
													  "int entradatx, int entradaty,\n"
													  "int saidatx, int saidaty,\n"
													  "int k0");

	} else if (type_pooling == AVEPOOL) {
		KRN_new(self->poolativa, "poolAVativa", "Vector entrada, Vector saida,\n"
												"int passox,int passoy,\n"
												"int filtrox,int filtroy,\n"
												"int saidatx, int saidaty,\n"
												"int entradatx, int entradaty, int k0");

		KRN_new(self->poolCalcGrads, "poolAvCalcGrads", "Vector entrada, Vector gradEntrada,\n"
														"Vector gradNext, Vector saida,\n"
														"int fx, int fy, int px, int py,\n"
														"int entradatx, int entradaty,\n"
														"int saidatx, int saidaty,\n"
														"int k0");

	} else if (type_pooling == MINPOOL) {
		KRN_new(self->poolativa, "poolativaMin", "Vector entrada, Vector saida,\n"
												 "int passox,int passoy,\n"
												 "int filtrox,int filtroy,\n"
												 "int saidatx, int saidaty,\n"
												 "int entradatx, int entradaty, int k0");


		KRN_new(self->poolCalcGrads, "poolCalcGrads", "Vector entrada, Vector gradEntrada,\n"
													  "Vector gradNext, Vector saida,\n"
													  "int fx, int fy, int px, int py,\n"
													  "int entradatx, int entradaty,\n"
													  "int saidatx, int saidaty,\n"
													  "int k0");

	} else {
		ecx->setError(ecx, GAB_INVALID_PARAM);
		fprintf(stderr, "Tipo invalido\n");
		goto methods;
	}
	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaPooling_release;
	self->super.propagation = (int (*)(void *)) CamadaPooling_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaPooling_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaPooling_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaPooling_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaPooling_save;
	return (Camada) self;
}
