//
// Created by hslhe on 19/11/2021.
//

#include <camadas/funcoesDeAtivacao.h>
#include "camadas/CamadaConvNC.h"

static const char *lname = "ConvolucaoNC";

static void CamadaConvNC_release(CamadaConvNC *self_p) {
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
	Release((*self_p)->convncSum);
	Release((*self_p)->convncCalcFiltro);
	Release((*self_p)->convncCalcGradZ);
	Release((*self_p)->convncCalcGrads);
	Release((*self_p)->convncCalcFiltroBatch);
	Release((*self_p)->kernel_fixW);
	gab_free(*self_p);
	*self_p = NULL;
}

static Tensor  CamadaConvNC_propagation(CamadaConvNC self,Tensor a) {
	self->super.a = a;
	Execute(convncSum, self->super.s->length, &self->W->data, &self->super.a->data, &self->z->data, &self->super.s->data, &self->passox, &self->passoy, &self->activationFuntion, &self->passox, &self->passoy, &self->aberturax, &self->aberturay, &self->super.a->x, &self->super.a->y, &self->super.s->x, &self->super.s->y, &self->W->x, &self->W->y, &self->W->z);
	return self->super.s;
}

static int CamadaConvNC_backpropagation(CamadaConvNC self, Tensor ds) {
	if (self->super.da || !self->super.params.skipLearn) {
		Execute(convncSum, self->super.s->length, &ds->data, &self->z->data, &self->dz->data, &self->derivationFunction);
	}
	if (self->super.da) {
		Execute(convncCalcGrads, self->super.da->length,

				&self->W->data, &self->super.da->data, &self->dz->data, &self->passox, &self->passoy, &self->super.da->x, &self->super.da->y, &self->super.s->x, &self->super.s->y, &self->super.s->z, &self->W->x, &self->W->y, &self->W->z);
	}
	if (!self->super.params.skipLearn) {
		Execute(convncCalcFiltro, self->W->length, &self->dz->data, &self->super.a->data, &self->W->data, &self->dW->data, &self->dW->x, &self->dW->y, &self->dW->z, &self->super.a->x, &self->super.a->y, &self->super.s->x, &self->super.s->y, &self->passox, &self->passoy, &self->aberturax, &self->aberturay, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	}
	return self->super.ecx->error;
}

static int CamadaConvNC_backpropagationBatch(CamadaConvNC self, Tensor ds, size_t bacthSize) {
	if (self->super.da || !self->super.params.skipLearn) {
		Execute(convncSum, self->super.s->length, &ds->data, &self->z->data, &self->dz->data, &self->derivationFunction);
	}
	if (self->super.da) {
		Execute(convncCalcGrads, self->super.da->length, &self->W->data, &self->super.da->data, &self->dz->data, &self->passox, &self->passoy, &self->super.da->x, &self->super.da->y, &self->super.s->x, &self->super.s->y, &self->super.s->z, &self->W->x, &self->W->y, &self->W->z);
	}
	if (!self->super.params.skipLearn) {
		Execute(convncCalcFiltroBatch, self->W->length, &self->dz->data, &self->super.a->data, &self->dW->data, &bacthSize, &self->dW->x, &self->dW->y, &self->dW->z, &self->super.a->x, &self->super.a->y, &self->super.s->x, &self->super.s->y, &self->passox, &self->passoy, &self->aberturax, &self->aberturay);
	}
	return self->super.ecx->error;
}

static int CamadaConvNC_learnBatch(CamadaConvNC self) {
	if (!self->super.params.skipLearn) {
		//convncFixBatch(Vector w, Vector dw,REAL hitlearn, REAL momento, REAL weightDecay, int k0)
		Execute(kernel_fixW, self->W->length, &self->W->data, &self->dW->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	}
	return self->super.ecx->error;
}

static char *CamadaConvNC_json(CamadaConvNC self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"functionActivation\":%d,\n"
			PAD"\"passo\":[%zu,%zu],\n"
			PAD"\"largura\":[%zu,%zu],\n"
			PAD"\"numero_filtros\":%zu", tmp, self->activationFuntion, self->passox, self->passoy, self->aberturax, self->aberturay, self->W->w);

	apendTensor("w", W, string, len, tmp, showValues);
	apendTensor("dw", dW, string, len, tmp, showValues);
	apendTensor("Z", z, string, len, tmp, showValues);
	apendTensor("dZ", dz, string, len, tmp, showValues);

	apendstr(string, len, "\n}");

	return string;
}

static char *CamadaConvNC_getGenerate(CamadaConvNC self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s(P2D(%zu,%zu),P2D(%zu,%zu),P3D(%zu,%zu,%zu),%s,Params(%g,%g,%g,%d),RDP(%d,%g,%g))", lname, self->passox, self->passoy, self->aberturax, self->aberturay, self->W->x, self->W->y, self->W->w, F_ATIVACAO_NAME(self->activationFuntion), (double) self->super.params.hitlearn, (double) self->super.params.momento, (double) self->super.params.decaimento, self->super.params.skipLearn, self->rdp_filtros.type, (double) self->rdp_filtros.a, (double) self->rdp_filtros.b);
	return string;
}

/**
 * Salva a camada conv em um arquivo
 * camada
 * passo x,y
 * abertura x,y
 * filtro x,y,w
 * função de ativação
 * filtros
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
static int CamadaConvNC_save(CamadaConvNC self, FILE *f) {
	if (self->super.ecx->error) {
		goto end;
	}
	self->super.ecx->addstack(self->super.ecx, "CamadaConvF_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->aberturax, 1, sizeof(size_t), f);
	fwrite(&self->aberturay, 1, sizeof(size_t), f);
	fwrite(&self->W->x, 1, sizeof(size_t), f);
	fwrite(&self->W->y, 1, sizeof(size_t), f);
	fwrite(&self->W->w, 1, sizeof(size_t), f);
	fwrite(&self->activationFuntion, 1, sizeof(uint32_t), f);
	internal_saveTensor(f, self->W);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaConvNC_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaConvF_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;
	uint32_t fativacao;
	P2d passo;
	P2d abertura;
	P3d filtro;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&passo.x, sizeof(size_t), 1, f);
	fread(&passo.y, sizeof(size_t), 1, f);
	fread(&abertura.x, sizeof(size_t), 1, f);
	fread(&abertura.y, sizeof(size_t), 1, f);
	fread(&filtro.x, sizeof(size_t), 1, f);
	fread(&filtro.y, sizeof(size_t), 1, f);
	fread(&filtro.z, sizeof(size_t), 1, f);
	fread(&fativacao, sizeof(uint32_t), 1, f);

	CamadaConvNC self = (CamadaConvNC) CamadaConvNC_new(gpu, queue, passo, abertura, filtro, size_in, fativacao, entrada, parametros, ecx, RDP(-1));
	internal_loadTensor(f, self->W, size_element);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}
int CamadaConvNC_fprintf(CamadaConvNC self, FILE * destino, char *format, ...){
	va_list  v;
	va_start(v,format);
	internal_Camada_fprint(self,destino,format,v);
	fprintf(destino,"w -> ");self->W->fprint(self->W,destino);
	fprintf(destino,"dw -> ");self->dW->fprint(self->dW,destino);
	return 0;
}
Camada CamadaConvNC_new(Gpu gpu, Queue queue, P2d passo, P2d abertura, P3d filtro, P3d size_in, uint32_t ativacao, Tensor entrada, Parametros params, Ecx ecx, RandomParams rdp_filtros) {
	ECXPUSH(ecx);
	CamadaConvNC self = gab_alloc(1, sizeof(CamadaConvNC_t));
	P3d size_out = {(size_in.x - 1 - (filtro.x - 1) * abertura.x) / passo.x + 1, (size_in.y - 1 - (filtro.y - 1) * abertura.y) / passo.y + 1, filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAONC_ID, lname, params, entrada, size_in, size_out, ecx);
	self->activationFuntion = ativacao;
	self->derivationFunction = ativacao | FLAGDIF;
	self->dW = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->dW->fill(self->dW, 0);
	self->W = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	if (ecx->error) {
		goto methods;
	}
	self->rdp_filtros = rdp_filtros;
	if (rdp_filtros.type != -1) {
		if (rdp_filtros.type == 0) {
			rdp_filtros = internal_getDefaultRDP(ativacao == FRELU, size_in.x * size_in.y * size_in.z, self->super.s->length);
		}
		self->super.ecx->error = self->W->randomize(self->W, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);
		if (ecx->error) {
			goto methods;
		}

	}

	self->passox = passo.x;
	self->passoy = passo.y;
	self->aberturax = abertura.x;
	self->aberturay = abertura.y;
	KRN_news(self->convncSum, "convncSum", "Vector w, Vector A, Vector Z, Vector S, unsigned int fid, unsigned int passox, int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0");
	KRN_news(self->convncCalcGradZ, "convncCalcGradZ", "Vector ds, Vector z, Vector dz, unsigned int fid, int k0");
	KRN_news(self->convncCalcGrads, "convncCalcGrads", "Vector w, Vector DA, Vector dz, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0");
	KRN_news(self->convncCalcFiltro, "convncCalcFiltro", "Vector dz, Vector A, Vector w, Vector dw, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, REAL hitlearn, REAL momento, REAL weightDecay, int k0");
	KRN_news(self->convncCalcFiltroBatch, "convncCalcFiltroBatch", "Vector dz, Vector A, Vector dw,long batchSize, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, int k0");
	KRN_news(self->kernel_fixW, "kernel_fixW", "Vector w, Vector dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0");

	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConvNC_release;
	self->super.propagation = (Tensor (*)(void *, Tensor)) CamadaConvNC_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaConvNC_backpropagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaConvNC_backpropagationBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) CamadaConvNC_learnBatch;
	self->super.json = (char *(*)(void *, int)) CamadaConvNC_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConvNC_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaConvNC_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaConvNC_fprintf;
	return (Camada) self;
}


