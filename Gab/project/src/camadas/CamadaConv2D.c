//
// Created by hslhe on 19/11/2021.
//


#include "camadas/CamadaConv2D.h"

static const char *lname = "Convolucao2D";

static void CamadaConv2D_release(CamadaConv2D *self_p) {
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
	Release((*self_p)->conv2dSum);
	Release((*self_p)->conv2dCalcGradIn);
	Release((*self_p)->conv2dCalcGradZ);
	Release((*self_p)->conv2dCalcGradAndFixWeight);
	Release((*self_p)->conv2dCalcGradBatch);
	Release((*self_p)->kernel_fixW);
	gab_free(*self_p);
	*self_p = NULL;
}

static int CamadaConv2D_propagation(CamadaConv2D self) {
	//Vr W, Vr a, Vw Z, Vw s, int px, int py, int sx, int sy, int ax, int ay, int az, int fx, int fy, int fz, int fid
	Execute(conv2dSum, self->super.s->length, &self->W->data, &self->super.a->data, &self->z->data, &self->super.s->data, &self->passox, &self->passoy, &self->super.s->x, &self->super.s->y, &self->super.a->x, &self->super.a->y, &self->super.a->z, &self->W->x, &self->W->y, &self->W->z, &self->fid);
	return self->super.ecx->error;
}

static int CamadaConv2D_backpropagation(CamadaConv2D self, Tensor ds) {
	if (self->super.da || !self->super.params.skipLearn) {
		//Vr ds, Vr z, Vw dz, int fid
		Execute(conv2dCalcGradZ, self->super.s->length, &ds->data, &self->z->data, &self->dz->data, &self->dfid);
	}
	if (self->super.da) {
		//Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int az, int sx, int sy
		Execute(conv2dCalcGradIn, self->super.da->length, &self->W->data, &self->super.da->data, &self->dz->data, &self->W->x, &self->W->y, &self->W->z, &self->passox, &self->passoy, &self->super.da->x, &self->super.da->y, &self->super.da->z, &self->super.s->x, &self->super.s->y);
	}
	if (!self->super.params.skipLearn) {
		//Vrw W, Vr dz, Vr a, Vrw dW, int fx, int fy, int ax, int ay, int az, int sx, int sy, int px, int py, REAL hitLearn, REAL momento, REAL weightDecay
		Execute(conv2dCalcGradAndFixWeight, self->W->length, &self->W->data, &self->dz->data, &self->super.a->data, &self->dW->data, &self->W->x, &self->W->y, &self->super.a->x, &self->super.a->y, &self->super.a->y, &self->super.s->x, &self->super.s->y, &self->passox, &self->passoy, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
	}
	return self->super.ecx->error;
}

static int CamadaConv2D_backpropagationBatch(CamadaConv2D self, Tensor ds, size_t batchSize) {
	if (self->super.da || !self->super.params.skipLearn) {
		//Vr ds, Vr z, Vw dz, int fid, int k0
		Execute(conv2dCalcGradZ, self->super.s->length, &ds->data, &self->z->data, &self->dz->data, &self->dfid);
	}
	if (self->super.da) {
		//Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int az, int sx, int sy
		Execute(conv2dCalcGradIn, self->super.da->length, &self->W->data, &self->super.da->data, &self->dz->data, &self->W->x, &self->W->y, &self->W->z, &self->passox, &self->passoy, &self->super.da->x, &self->super.da->y, &self->super.da->z, &self->super.s->x, &self->super.s->y);
	}
	if (!self->super.params.skipLearn) {
//		printf("0x%llX,%s:%d\n",(unsigned  long long int)self,__FILE__,__LINE__);
//		self->W->fprint(self->W,stdout);
		//Vr dz, Vr a, Vr dW, long batchSize, int fx, int fy, int fz, int ax, int ay, int az, int sx, int sy, int px, int py
		Execute(conv2dCalcGradBatch, self->W->length, &self->dz->data, &self->super.a->data, &self->dW->data, &batchSize, &self->W->x, &self->W->y, &self->W->z, &self->super.a->x, &self->super.a->y, &self->super.a->z, &self->super.s->x, &self->super.s->y, &self->passox, &self->passoy);
//		self->W->fprint(self->W,stdout);

	}
	return self->super.ecx->error;
}

static int CamadaConv2D_learnBatch(CamadaConv2D self) {
	if (!self->super.params.skipLearn) {
//		printf("0x%llX,%s:%d\n",(unsigned  long long int)self,__FILE__,__LINE__);
//		self->W->fprint(self->W,stdout);
//		fflush(stdout);
//		printf("%f %f %f\n",self->super.params.hitlearn,self->super.params.momento,self->super.params.decaimento);
		Execute(kernel_fixW, self->W->length, &self->W->data, &self->dW->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
//		self->W->fprint(self->W,stdout);
//		fflush(stdout);

	}
	return self->super.ecx->error;
}

static char *CamadaConv2D_json(CamadaConv2D self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"functionActivation\":%d,\n"
			PAD"\"passo\":[%zu,%zu],\n"
			PAD"\"numero_filtros\":%zu", tmp, self->fid, self->passox, self->passoy, self->W->w);
	gab_free(tmp);
	apendTensor("filtros", W, string, len, tmp, showValues);
	apendTensor("grad_filtros", dW, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaConv2D_getGenerate(CamadaConv2D self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s (P2D(%zu, %zu), P3D(%zu, %zu, %zu), %s, Params(%g, %g, %g, %d), RDP(%d, %g, %g))", lname, self->passox, self->passoy, self->W->x, self->W->y, self->W->z, F_ATIVACAO_NAME(self->fid), (double) self->super.params.hitlearn, (double) self->super.params.momento, (double) self->super.params.decaimento, self->super.params.skipLearn, self->rdp_filtros.type, (double) self->rdp_filtros.a, (double) self->rdp_filtros.b);

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
static int CamadaConv2D_save(CamadaConv2D self, FILE *f) {
	if (self->super.ecx->error) {
		goto end;
	}
	self->super.ecx->addstack(self->super.ecx, "CamadaConv2D_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->W->x, 1, sizeof(size_t), f);
	fwrite(&self->W->y, 1, sizeof(size_t), f);
	fwrite(&self->W->w, 1, sizeof(size_t), f);
	fwrite(&self->fid, 1, sizeof(uint32_t), f);
	internal_saveTensor(f, self->W);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaConv2D_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaConv2D_load");
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

	CamadaConv2D self = (CamadaConv2D) CamadaConv2D_new(gpu, queue, passo, filtro, size_in, fativacao, entrada, parametros, ecx, RDP(-1));
	internal_loadTensor(f, self->W, size_element);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}

int CamadaConv2D_fprintf(CamadaConv2D self, FILE *destino, char *format, ...) {
	va_list v;
	va_start(v, format);
	internal_Camada_fprint(self, destino, format, v);
	fprintf(destino, "W -> ");
	self->W->fprint(self->W, destino);
	fprintf(destino, "dW -> ");
	self->dW->fprint(self->dW, destino);
	fprintf(destino, "z -> ");
	self->z->fprint(self->z, destino);
	fprintf(destino, "dz -> ");
	self->dz->fprint(self->dz, destino);
	return 0;
}

Camada CamadaConv2D_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, uint32_t fid, Tensor entrada, Parametros params, Ecx ecx, RandomParams rdp_filtros) {
	ECXPOP(ecx);
	CamadaConv2D self = gab_alloc(1, sizeof(CamadaConv2D_t));
	P3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1, filtro.z * size_in.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAO2D_ID, lname, params, entrada, size_in, size_out, ecx);
	self->fid = fid;
	self->dfid = fid | FLAGDIF;
	self->W = Tensor_new(filtro.x, filtro.y, filtro.z, 1, ecx, 0, gpu->context, queue);
	self->dW = Tensor_new(filtro.x, filtro.y, filtro.z, 1, ecx, 0, gpu->context, queue);
	self->dW->fill(self->dW, 0);
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	if (ecx->error) {
		goto methods;
	}
	self->rdp_filtros = rdp_filtros;
	if (rdp_filtros.type != -1) {
		if (rdp_filtros.type == 0) {
			rdp_filtros = internal_getDefaultRDP(fid == FRELU, size_in.x * size_in.y * size_in.z, self->super.s->length);
		}
		self->super.ecx->error = self->W->randomize(self->W, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);
		if (ecx->error) {
			goto methods;
		}
	}
	self->passox = passo.x;
	self->passoy = passo.y;
	Knew_conv2dSum(self->conv2dSum);
	Knew_conv2dCalcGradZ(self->conv2dCalcGradZ);
	Knew_conv2dCalcGradIn(self->conv2dCalcGradIn);
	Knew_conv2dCalcGradAndFixWeight(self->conv2dCalcGradAndFixWeight);
	Knew_conv2dCalcGradBatch(self->conv2dCalcGradBatch);
	Knew_kernel_fixW(self->kernel_fixW);

	ECXPOP(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConv2D_release;
	self->super.propagation = (int (*)(void *)) CamadaConv2D_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaConv2D_backpropagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaConv2D_backpropagationBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) CamadaConv2D_learnBatch;
	self->super.json = (char *(*)(void *, int)) CamadaConv2D_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConv2D_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaConv2D_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaConv2D_fprintf;
	return (Camada) self;
}


