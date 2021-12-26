//
// Created by hslhe on 14/11/2021.
//

#include "camadas/CamadaConv.h"

static const char *lname = "Convolucao";

static void CamadaConv_release(CamadaConv *self_p) {
	if (!self_p) { return; }
	if (!*self_p) { return; }
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->W);
	Release((*self_p)->dW);
	Release((*self_p)->convSum);
	Release((*self_p)->convCalcGradIn);
	Release((*self_p)->convCalcGradAndFixWeight);
	gab_free(*self_p);
	*self_p = NULL;
}

static int CamadaConv_propagation(CamadaConv self) {
	Execute(convSum, self->super.s->length,

			&self->W->data, &self->super.a->data, &self->super.s->data, &self->passox, &self->passoy, &self->super.s->x, &self->super.s->y, &self->super.a->x, &self->super.a->y, &self->W->x, &self->W->y, &self->W->z);
	return self->super.ecx->error;
}

static int CamadaConv_backpropagation(CamadaConv self, Tensor ds) {
	if (self->super.da)
		Execute(convCalcGradIn, self->super.da->length,

				&self->W->data, &self->super.da->data, &ds->data, &self->W->x, &self->W->y, &self->W->z, &self->passox, &self->passoy, &self->super.da->x, &self->super.da->y, &self->super.s->x, &self->super.s->y, &self->super.s->z);
	if (!self->super.params.skipLearn)
		Execute(convCalcGradAndFixWeight, self->W->length, &self->W->data, &ds->data, &self->super.a->data, &self->dW->data, &self->W->x, &self->W->y, &self->W->z, &self->super.a->x, &self->super.a->y, &self->super.s->x, &self->super.s->y, &self->passox, &self->passoy, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	return self->super.ecx->error;
}

static char *CamadaConv_json(CamadaConv self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{\n"
			PAD"%s,\n"
			PAD"\"passo\":[%zu,%zu],\n"
			PAD"\"numero_filtros\":%zu", tmp, self->passox, self->passoy, self->W->w);
	gab_free(tmp);
	apendTensor("filtros", W, string, len, tmp, showValues);
	apendTensor("grad_filtros", dW, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaConv_getGenerate(CamadaConv self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s(P2D(%zu,%zu),P3D(%zu,%zu,%zu),Params(%g,%g,%g,%d),RDP(%d,%g,%g))", lname, self->passox, self->passoy, self->W->x, self->W->y, self->W->w, (double) self->super.params.hitlearn, (double) self->super.params.momento, (double) self->super.params.decaimento, self->super.params.skipLearn, self->rdp_filtros.type, (double) self->rdp_filtros.a, (double) self->rdp_filtros.b);

	return string;
}

/**
 * Salva a camada conv em um arquivo
 * camada
 * passo x e y
 * filtro x,y,w
 * filtros
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
static int CamadaConv_save(CamadaConv self, FILE *f) {
	if (self->super.ecx->error) { goto end; }
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->passox, sizeof(size_t), 1, f);
	fwrite(&self->passoy, sizeof(size_t), 1, f);
	fwrite(&self->W->x, sizeof(size_t), 1, f);
	fwrite(&self->W->y, sizeof(size_t), 1, f);
	fwrite(&self->W->w, sizeof(size_t), 1, f);
	internal_saveTensor(f, self->W);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaConv_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaConv_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;
	P2d passo;
	P3d filtro;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&passo.x, sizeof(size_t), 1, f);
	fread(&passo.y, sizeof(size_t), 1, f);
	fread(&filtro.x, sizeof(size_t), 1, f);
	fread(&filtro.y, sizeof(size_t), 1, f);
	fread(&filtro.z, sizeof(size_t), 1, f);
	CamadaConv self = (CamadaConv) CamadaConv_new(gpu, queue, size_in, entrada, ecx, passo, filtro, parametros, RDP(-1));
	internal_loadTensor(f, self->W, size_element);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}

Camada CamadaConv_new(INTERNAL_DEFAULT_ARGS, P2d passo, P3d filtro, Parametros params, RandomParams rdp_filtros) {
	ecx->addstack(ecx, "CamadaConv_new");
	CamadaConv self = gab_alloc(1, sizeof(CamadaConv_t));
	P3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1, filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAO_ID, lname, params, entrada, size_in, size_out, ecx);

	self->dW = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->dW->fill(self->dW, 0);
	self->W = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	if (ecx->error) { goto methods; }
	self->rdp_filtros = rdp_filtros;
	if (rdp_filtros.type != -1) {
		if (rdp_filtros.type == 0) {
			rdp_filtros = internal_getDefaultRDP(0, size_in.x * size_in.y * size_in.z, self->super.s->length);
		}

		self->super.ecx->error = self->W->randomize(self->W, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);
		if (ecx->error) { goto methods; }

	}

	self->passox = passo.x;
	self->passoy = passo.y;
	KRN_new(self->convSum, "convSum", "Vector filtro, Vector entrada, Vector saida,int passox, int passoy,int saidatx, int saidaty,int entradatx, int entradaty,int fx, int fy, int fz, int k0")

	KRN_new(self->convCalcGradAndFixWeight, "convCalcGradAndFixWeight", "Vector filtros, Vector ds, Vector entrada, Vector gradFiltro,int fx, int fy, int fz,int entrada_tx, int entrada_ty,int saida_tx, int saida_ty,int passox, int passoy,REAL hitLearn, REAL momento, REAL weightDecay,int k0")

	KRN_new(self->convCalcGradIn, "convCalcGradIn", "Vector filtro, Vector gradEntrada, Vector gradNext,int fx, int fy, int fz,int passox, int passoy,int entradatx, int entradaty,int saidatx, int saidaty, int saidatz,int k0")
	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConv_release;
	self->super.propagation = (int (*)(void *)) CamadaConv_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaConv_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaConv_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConv_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaConv_save;
	return (Camada) self;
}
