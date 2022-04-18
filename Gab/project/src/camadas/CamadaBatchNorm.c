//
// Created by hslhe on 18/11/2021.
//

#include "camadas/CamadaBatchNorm.h"


static const char *lname = "BatchNorm";

void CamadaBatchNorm_release(CamadaBatchNorm *selfp) {
	internal_Camada_release((Camada *) (selfp));
	Release((*selfp)->Y);
	Release((*selfp)->dY);
	Release((*selfp)->B);
	Release((*selfp)->dB);
	Release((*selfp)->media);
	Release((*selfp)->inv_std);
	Release((*selfp)->norma);
	Release((*selfp)->dnorma);
	Release((*selfp)->media_dnorma);
	Release((*selfp)->media_dnorma_norma);

	Release((*selfp)->BatchNormMedia);
	Release((*selfp)->BatchNormInvDesv);
	Release((*selfp)->BatchNormNormaliza);
	Release((*selfp)->BatchNormaCalcDnorm);
	Release((*selfp)->BatchNormMediadnorm_norma);
	Release((*selfp)->BatchNormMediadnorm_norma);
	Release((*selfp)->BatchNormaCalcDa);
	Release((*selfp)->BatchNormaCalcdYdB);
	Release((*selfp)->BatchNormaLearn);

	gab_free(*selfp);
}

/***
 *  Passos para fazer na batchnorm\n
 *  1 - achar a média dos dados
 *  2 - achar a variância e o inverso do desvio padrão
 *  3 - normalizar
 *  4 - multiplicar pela nova variância e pela média
 * @param self
 * @return
 */
Tensor CamadaBatchNorm_propagation(CamadaBatchNorm self, Tensor a) {

	// acha a média aqui da entrada
	Execute(BatchNormMedia, self->super.s->z, a->data, &self->media->data, &self->super.a->x, &self->super.a->y);

	// acha a variancia e inverso do desvio
	Execute(BatchNormInvDesv, self->super.s->z, &self->super.a->data, &self->media->data, &self->inv_std->data, &self->epsilon, &self->super.s->x, &self->super.s->y);

	// normalizar entrada
	//Vector saida, Vector norma, Vector a, Vector media, Vector inv_std, Vector Y, Vector b, int ax, int ay, int k0
	Execute(BatchNormNormaliza, self->super.s->length, &self->super.s->data, &self->norma->data, &self->super.a->data, &self->media->data, &self->inv_std->data, &self->Y->data, &self->B->data, &self->super.s->x, &self->super.s->y);

	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return self->super.s;
}

int CamadaBatchNorm_retroPropagationBatch(CamadaBatchNorm self, Tensor ds, size_t batchSize) {
	if (self->super.da || !self->super.params.skipLearn) { //  calcular o gradiente da norma ds * dY
		Execute(BatchNormaCalcDnorm, self->super.s->length, &self->dnorma->data, &ds->data, &self->Y->data, &self->super.a->x, &self->super.a->y);
	}

	if (self->super.da) {// calcular o gradiente de entrada
		Execute(BatchNormMediadnorm_norma, self->super.s->z, &self->norma->data, &self->dnorma->data, &self->media_dnorma->data, &self->media_dnorma_norma->data, &self->super.s->x, &self->super.s->y);
		//Vector da, Vector norm, Vector dnorm, Vector mdnorm, Vector mdnormnorm, Vector inv_std, int ax, int ay,
		Execute(BatchNormaCalcDa, self->super.da->length, &self->super.da->data, &self->norma->data, &self->dnorma->data, &self->media_dnorma->data, &self->media_dnorma_norma->data, &self->inv_std->data, &self->super.a->x, &self->super.a->y);
	}

	if (!self->super.params.skipLearn) {
//		Vector ds, Vector norma, Vector gradY, Vector gradB, long batchSize, int ax, int ay, int k0
		Execute(BatchNormaCalcdYdB, self->super.s->z, &ds->data, &self->norma->data, &self->dY->data, &self->dB->data, &batchSize, &self->super.s->x, &self->super.s->y);
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return self->super.ecx->error;
}

int CamadaBatchNorm_backpropagation(CamadaBatchNorm self, Tensor ds) {
	CamadaBatchNorm_retroPropagationBatch(self, ds, self->batch_size);
	if (!self->super.params.skipLearn) {
		self->batch++;
		if (self->batch >= self->batch_size) {
			self->batch = 0;
			Execute(BatchNormaLearn, self->Y->length, &self->Y->data, &self->B->data, &self->dY->data, &self->dB->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
		}
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return self->super.ecx->error;
}

int CamadaBatchNorm_retroPropagationBatchLearn(CamadaBatchNorm self) {
	if (!self->super.params.skipLearn) {
		Execute(BatchNormaLearn, self->Y->length, &self->Y->data, &self->B->data, &self->dY->data, &self->dB->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return self->super.ecx->error;
}

char *CamadaBatchNorm_json(CamadaBatchNorm self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"epsilon\":%g", tmp, (double) self->epsilon);
	gab_free(tmp);
	apendTensor("Y", Y, string, len, tmp, showValues);
	apendTensor("dY", dY, string, len, tmp, showValues);
	apendTensor("b", B, string, len, tmp, showValues);
	apendTensor("dB", dB, string, len, tmp, showValues);
	apendTensor("media", media, string, len, tmp, showValues);
	apendTensor("norma", norma, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

char *CamadaBatchNorm_getGenerate(CamadaBatchNorm self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s(%lld,%g,Params(%g,%g,%g,%d),RDP(%d,%g,%g),RDP(%d,%g,%g))", lname, self->batch_size, self->epsilon, (double) self->super.params.hitlearn, (double) self->super.params.momento, (double) self->super.params.decaimento, self->super.params.skipLearn, self->rdp_Y.type, (double) self->rdp_Y.a, (double) self->rdp_Y.b, self->rdp_B.type, (double) self->rdp_B.a, (double) self->rdp_B.b);
	return string;
}

/**
 * Salva a camada batchnorm em um arquivo
 * Camada
 * epsilon
 * Y
 * b
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
int CamadaBatchNorm_save(CamadaBatchNorm self, FILE *f) {
	ECX_RETURN_IF_ERROR(Super.ecx, Super.ecx->error)
	internal_saveCamada(f, (Camada) self);
	internal_saveREAL(f, self->epsilon);
	internal_saveTensor(f, self->Y);
	internal_saveTensor(f, self->B);
	fwrite(&self->batch_size, sizeof(size_t), 1, f);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return self->super.ecx->error;
}

Camada CamadaBatchNorm_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ECX_RETURN_IF_ERROR(ecx, NULL)

	P3d size_in;
	Parametros parametros;
	uint32_t size_element;
	size_t batchsize = 1;
	REAL epsilon;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	internal_loadREAL(f, &epsilon, size_element);
	fread(&batchsize, sizeof(size_t), 1, f);
	CamadaBatchNorm self = (CamadaBatchNorm) CamadaBatchNorm_new(gpu, queue, size_in, entrada, ecx, parametros, epsilon, batchsize, RDP(-1), RDP(-1));
	internal_loadTensor(f, self->Y, size_element);
	internal_loadTensor(f, self->B, size_element);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return (Camada) self;
}

int CamadaBatchNorm_fprintf(CamadaBatchNorm self, FILE *destino, char *format, ...) {
	va_list v;
	va_start(v, format);
	internal_Camada_fprint(self, destino, format, v);
	fprintf(destino, "Y -> ");
	self->Y->fprint(self->Y, destino);
	fprintf(destino, "dY -> ");
	self->dY->fprint(self->dY, destino);
	fprintf(destino, "b -> ");
	self->B->fprint(self->B, destino);
	fprintf(destino, "dB -> ");
	self->dB->fprint(self->dB, destino);
	fprintf(destino, "media -> ");
	self->media->fprint(self->media, destino);
	fprintf(destino, "inv_std -> ");
	self->inv_std->fprint(self->inv_std, destino);
	fprintf(destino, "norma -> ");
	self->norma->fprint(self->norma, destino);
	fprintf(destino, "dnorma -> ");
	self->dnorma->fprint(self->dnorma, destino);
	fprintf(destino, "media_dnorma -> ");
	self->media_dnorma->fprint(self->media_dnorma, destino);
	fprintf(destino, "media_dnorma_norma -> ");
	self->media_dnorma_norma->fprint(self->media_dnorma_norma, destino);
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return 0;
}

extern Camada CamadaBatchNorm_new(INTERNAL_DEFAULT_ARGS, Parametros params, REAL epsilon, size_t batchSize, Rdp randY, Rdp randB) {
	ECX_RETURN_IF_ERROR(ecx, NULL)
	CamadaBatchNorm self = gab_alloc(1, sizeof(CamadaBatchNorm_t));

	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, BATCHNORM_ID, lname, params, entrada, size_in, size_out, ecx);
	ECX_IF_FAILED(Super.ecx, methods)
	self->batch_size = batchSize;

	self->Y = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->dY = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->B = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->dB = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->media = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->inv_std = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->norma = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->dnorma = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->media_dnorma = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->media_dnorma_norma = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);

	self->epsilon = epsilon;
	ECX_IF_FAILED(Super.ecx, methods)
	self->dY->fill(self->dY, 0);
	self->dB->fill(self->dB, 0);
	ECX_IF_FAILED(Super.ecx, methods)
	//kernels
	Knew_BatchNormMedia(self->BatchNormMedia);
	Knew_BatchNormInvDesv(self->BatchNormInvDesv);
	Knew_BatchNormNormaliza(self->BatchNormNormaliza);
	Knew_BatchNormaCalcDnorm(self->BatchNormaCalcDnorm);
	Knew_BatchNormMediadnorm_norma(self->BatchNormMediadnorm_norma);
	Knew_BatchNormaCalcDa(self->BatchNormaCalcDa);
	Knew_BatchNormaCalcdYdB(self->BatchNormaCalcdYdB);
	Knew_BatchNormaLearn(self->BatchNormaLearn);

	self->rdp_Y = randY;
	self->rdp_B = randB;

	if (randY.type != -1) {
		if (randY.type == 0) {
			randY = internal_getDefaultRDP(1, size_in.x * size_in.y * size_in.z, self->super.s->length);
			self->rdp_Y = randY;
			self->rdp_Y.type = 0;
		}
		self->super.ecx->setError(self->super.ecx, self->Y->randomize(self->Y, randY.type, randY.a, randY.b), "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);

	}

	if (randB.type != -1) {
		if (randB.type == 0) {
			randB = internal_getDefaultRDP(1, size_in.x * size_in.y * size_in.z, self->super.s->length);
		}
		self->super.ecx->setError(self->super.ecx, self->B->randomize(self->B, randB.type, randB.a, randB.b), "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
	}
	methods:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	self->super.release = (void (*)(void *)) CamadaBatchNorm_release;
	self->super.propagation = (Tensor (*)(void *, Tensor)) CamadaBatchNorm_propagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaBatchNorm_retroPropagationBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) CamadaBatchNorm_retroPropagationBatchLearn;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaBatchNorm_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaBatchNorm_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaBatchNorm_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaBatchNorm_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaBatchNorm_fprintf;
	return (Camada) self;
}
