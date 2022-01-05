//
// Created by hslhe on 19/11/2021.
//


#include "camadas/CamadaConvF.h"

static const char *lname = "ConvolucaoF";

static void CamadaConvF_release(CamadaConvF *self_p) {
	if (!self_p) {
		return;
	}
	if (!*self_p) {
		return;
	}
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->W);
	Release((*self_p)->dW);
	Release((*self_p)->z);
	Release((*self_p)->dz);
	Release((*self_p)->B);
	Release((*self_p)->dB);
	Release((*self_p)->convFSum);
	Release((*self_p)->convFCalcGradIn);
	Release((*self_p)->convFCalcGradBAndFix);
	Release((*self_p)->convFCalcGradBBatch);
	Release((*self_p)->convFCalcGradIn);
	Release((*self_p)->convFCalcGradZ);
	Release((*self_p)->convFCalcGradAndFixWeight);
	Release((*self_p)->convFCalcGradBatch);
	Release((*self_p)->kernel_fixW);
	gab_free(*self_p);
	*self_p = NULL;
}

static int CamadaConvF_propagation(CamadaConvF self) {
	KExec_convFSum(convFSum, self->super.s->length, self->W->data, self->B->data, self->super.a->data, self->z->data, self->super.s->data, self->passox, self->passoy, self->super.s->x, self->super.s->y, self->super.a->x, self->super.a->y, self->W->x, self->W->y, self->W->z, self->activationFuntion);
	return self->super.ecx->error;
}

static int CamadaConvF_backpropagation(CamadaConvF self, Tensor ds) {
	if (self->super.da || !self->super.params.skipLearn) {
		Execute(convFCalcGradZ, self->super.s->length, &ds->data, &self->z->data, &self->dz->data, &self->derivationFuntion);
	}
	if (self->super.da) {
		KExec_convFCalcGradIn(convFCalcGradIn, self->super.da->length, self->W->data, self->super.da->data, self->dz->data, self->W->x, self->W->y, self->W->z, self->passox, self->passoy, self->super.da->x, self->super.da->y, self->super.s->x, self->super.s->y, self->super.s->z);
	}
	if (!self->super.params.skipLearn) {
		KExec_convFCalcGradBAndFix(convFCalcGradBAndFix, self->dB->length, self->B->data, self->dB->data, self->dz->data, self->dz->x, self->dz->y, self->super.params.hitlearn, self->super.params.momento, self->super.params.decaimento);
		KExec_convFCalcGradAndFixWeight(convFCalcGradAndFixWeight, self->W->length, self->W->data, self->dz->data, self->super.a->data, self->dW->data, self->W->x, self->W->y, self->W->z, self->super.a->x, self->super.a->y, self->super.s->x, self->super.s->y, self->passox, self->passoy, self->super.params.hitlearn, self->super.params.momento, self->super.params.decaimento);
	}
	return self->super.ecx->error;
}

static int CamadaConvF_backpropagationBatch(CamadaConvF self, Tensor ds, size_t batchSize) {
	if (self->super.da || !self->super.params.skipLearn) {
		Execute(convFCalcGradZ, self->super.s->length, &ds->data, &self->z->data, &self->dz->data, &self->derivationFuntion);
	}
	if (self->super.da) {
		KExec_convFCalcGradIn(convFCalcGradIn, self->super.da->length, self->W->data, self->super.da->data, self->dz->data, self->W->x, self->W->y, self->W->z, self->passox, self->passoy, self->super.a->x, self->super.a->y, self->super.s->x, self->super.s->y, self->super.s->z);
	}
	if (!self->super.params.skipLearn) {
		KExec_convFCalcGradBBatch(convFCalcGradBBatch, self->dB->length, self->dB->data, self->dz->data, self->dz->x, self->dz->y, batchSize);
		KExec_convFCalcGradBatch(convFCalcGradBatch, self->dW->length, self->dz->data, self->super.a->data, self->dW->data, batchSize, self->W->x, self->W->y, self->W->z, self->super.a->x, self->super.a->y, self->super.s->x, self->super.s->y, self->passox, self->passoy);
	}
	return self->super.ecx->error;
}

static int CamadaConvF_learnBatch(CamadaConvF self) {
	if (!self->super.params.skipLearn) {
		KExec_kernel_fixW(kernel_fixW, self->W->length, self->W->data, self->dW->data, self->super.params.hitlearn, self->super.params.momento, self->super.params.decaimento);
		KExec_kernel_fixW(kernel_fixW, self->B->length, self->B->data, self->dB->data, self->super.params.hitlearn, self->super.params.momento, self->super.params.decaimento);
	}
	return self->super.ecx->error;
}

static char *CamadaConvF_json(CamadaConvF self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"functionActivation\":%d,\n"
			PAD"\"passo\":[%zu,%zu],\n"
			PAD"\"numero_filtros\":%zu", tmp, self->activationFuntion, self->passox, self->passoy, self->W->w);
	gab_free(tmp);
	apendTensor("filtros", W, string, len, tmp, showValues);
	apendTensor("grad_filtros", dW, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaConvF_getGenerate(CamadaConvF self) {
	char *string = NULL;
	int len = 0;
	GEN_LAYERNAME(string,len);
	GENN_P2D(P2D(self->passox,self->passoy),string,len);
	GENN_P3D(P3D(self->W->x,self->W->y,self->W->w),string,len);
	GENN_PARAMS(self->super.params,string,len);
	GEN_RDP(self->rdp_filtros,string,len);
	GEN_END(string,len);
//	apendstr(string, len, "%s (P2D(%zu, %zu), P3D(%zu, %zu, %zu), %s, Params(%g, %g, %g, %d), RDP(%d, %g, %g))", lname, self->passox, self->passoy, self->W->x, self->W->y, self->W->w, F_ATIVACAO_NAME(self->activationFuntion), (double) self->super.params.hitlearn, (double) self->super.params.momento, (double) self->super.params.decaimento, self->super.params.skipLearn, self->rdp_filtros.type, (double) self->rdp_filtros.a, (double) self->rdp_filtros.b);

	return string;
}

/**
 * Salva a camada conv em um arquivo
 * camada
 * passo x,y
 * filtro x,y,w
 * função de ativação
 * filtros
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
static int CamadaConvF_save(CamadaConvF self, FILE *f) {
	if (self->super.ecx->error) {
		goto end;
	}
	self->super.ecx->addstack(self->super.ecx, "CamadaConvF_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->W->x, 1, sizeof(size_t), f);
	fwrite(&self->W->y, 1, sizeof(size_t), f);
	fwrite(&self->W->w, 1, sizeof(size_t), f);
	fwrite(&self->activationFuntion, 1, sizeof(uint32_t), f);
	internal_saveTensor(f, self->W);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaConvF_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaConvF_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;
	uint32_t fativacao;
	P2d passo;
	P3d filtro;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&passo.x, sizeof(size_t), 1, f);
	fread(&passo.y, sizeof(size_t), 1, f);
	fread(&filtro.x, sizeof(size_t), 1, f);
	fread(&filtro.y, sizeof(size_t), 1, f);
	fread(&filtro.z, sizeof(size_t), 1, f);
	fread(&fativacao, sizeof(uint32_t), 1, f);

	CamadaConvF self = (CamadaConvF) CamadaConvF_new(gpu, queue, passo, filtro, size_in, fativacao, entrada, parametros, ecx, RDP(-1));
	internal_loadTensor(f, self->W, size_element);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}

int CamadaConvF_fprintf(CamadaConvF self, FILE *destino, char *format, ...) {
	va_list v;
	va_start(v, format);
	internal_Camada_fprint(self, destino, format, v);
	fprintf(destino, "W -> ");
	self->W->fprint(self->W, destino);
	fprintf(destino, "dW -> ");
	self->dW->fprint(self->dW, destino);
	return 0;
}

Camada CamadaConvF_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, FAtivacao_t ativacao, Tensor entrada, Parametros params, Ecx ecx, RandomParams rdp_filtros) {
	ECXPOP(ecx);
	CamadaConvF self = gab_alloc(1, sizeof(CamadaConvF_t));
	P3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1, filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAOF_ID, lname, params, entrada, size_in, size_out, ecx);
	self->activationFuntion.mask =  ativacao;
	if(self->activationFuntion.id == FRELU){
		self->activationFuntion.mask = FATIVACAO(FLRELU,0,1);
	}
	self->derivationFuntion = ativacao| FLAGDIF;
	self->dW = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->dW->fill(self->dW, 0);
	self->B = Tensor_new(1, 1, filtro.z, 1, ecx, TENSOR3D, gpu->context, queue);
	self->dB = Tensor_new(1, 1, filtro.z, 1, ecx, TENSOR3D, gpu->context, queue);
	self->B->fill(self->B, 0);
	self->B->fill(self->dB, 0);
	self->W = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	if (ecx->error) {
		goto methods;
	}
	self->rdp_filtros = rdp_filtros;
	if (rdp_filtros.type != -1) {
		if (rdp_filtros.type == 0) {
			rdp_filtros = internal_getDefaultRDP(ativacao == FLRELU, filtro.x * filtro.y * size_in.z, self->super.s->length);
		}
		self->super.ecx->error = self->W->randomize(self->W, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);
		if (ecx->error) {
			goto methods;
		}
	}
	self->passox = passo.x;
	self->passoy = passo.y;
	Knew_convFSum(self->convFSum);
	Knew_convFCalcGradZ(self->convFCalcGradZ);
	Knew_convFCalcGradBAndFix(self->convFCalcGradBAndFix);
	Knew_convFCalcGradBBatch(self->convFCalcGradBBatch);
	Knew_convFCalcGradIn(self->convFCalcGradIn);
	Knew_convFCalcGradAndFixWeight(self->convFCalcGradAndFixWeight);
	Knew_convFCalcGradBatch(self->convFCalcGradBatch);
	Knew_kernel_fixW(self->kernel_fixW);

	ECXPOP(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConvF_release;
	self->super.propagation = (int (*)(void *)) CamadaConvF_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaConvF_backpropagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaConvF_backpropagationBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) CamadaConvF_learnBatch;
	self->super.json = (char *(*)(void *, int)) CamadaConvF_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConvF_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaConvF_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaConvF_fprintf;
	return (Camada) self;
}


