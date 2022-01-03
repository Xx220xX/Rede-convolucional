//
// Created by Henrique on 19/11/2021.
//
/***
 * Camada fullconnect aplica as equações de uma rede profunda.
 * Condensando todas as dimensões da entrada em um unico vetor coluna  de dimensão(x*y*z,1,1)
 * A matriz de pesos (w) é da dimensão (n,x*y*z,1), onde n é o numero de neuronios de saída.
 * A matriz de bias (b) é da dimensão (n,1,1), onde n é o numero de neuronios de saída.
 *
 * Nesta implementação a matriz de saída é transposta, possuindo dimensão s' = (1,n,1)
 * @Propagation
 * A propagação é feita por
 * z = w*a + b
 * s = f(z)
 *
 * @Retropropagação
 * ds -> gradiente da saída
 * dz = f'(z)*ds
 * dw = dz*a
 * db = dz
 * da = w(T)*dz
 *
 */
#include "camadas/CamadaFullConnect.h"

static const char *lname = "FullConnect";

void CamadaFullConnect_release(CamadaFullConnect *self) {
	internal_Camada_release((Camada *) self);
	Release((*self)->w);
	Release((*self)->dw);
	Release((*self)->b);
	Release((*self)->db);
	Release((*self)->z);
	Release((*self)->dz);
	Release((*self)->fullfeed);
	Release((*self)->fullCalcDWandFix);
	Release((*self)->fullCalcDz);
	Release((*self)->fullCalcDzandFixB);
	Release((*self)->fullcalcin);
	Release((*self)->fullCalcDWBatch);
	Release((*self)->fullCalcDzBath);
	Release((*self)->kernel_fixW);

	gab_free(*self);
}

int CamadaFullConnect_propagation(CamadaFullConnect self) {
	Execute(fullfeed, self->super.s->length, &self->super.a->data, &self->w->data, &self->b->data, &self->z->data, &self->super.s->data, &self->fa, &self->w->x, &self->w->y);
	return self->super.ecx->error;
}

int CamadaFullConnect_backpropagation(CamadaFullConnect self, Tensor ds) {

	if (self->super.da || !self->super.params.skipLearn) {
		if (self->super.params.skipLearn) {
			Execute(fullCalcDz, self->dz->length, &self->dz->data, &ds->data, &self->z->data, &self->dfa);
		} else {
			Execute(fullCalcDzandFixB, self->dz->length, &self->dz->data, &ds->data, &self->z->data, &self->b->data, &self->db->data, &self->dfa, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento);
		}
		if (self->super.da) {
			Execute(fullcalcin, self->super.da->length, &self->dz->data, &self->super.da->data, &self->w->data, &self->w->x, &self->w->y);
		}
		if (!self->super.params.skipLearn) {
			Execute(fullCalcDWandFix, self->w->length, &self->super.a->data, &self->w->data, &self->dw->data, &self->dz->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento, &self->w->y);
		}
	}
	return self->super.ecx->error;

}

int CamadaFullConnect_backpropagationBatch(CamadaFullConnect self, Tensor ds, size_t batchSize) {

	if (self->super.da || !self->super.params.skipLearn) {

		Execute(fullCalcDzBath, self->dz->length, &self->dz->data, &ds->data, &self->z->data, &self->db->data, &self->dfa, &batchSize);
		if (self->super.da) {
			Execute(fullcalcin, self->super.da->length, &self->dz->data, &self->super.da->data, &self->w->data, &self->w->x, &self->w->y);
		}
		if (!self->super.params.skipLearn) {
			Execute(fullCalcDWBatch, self->w->length, &self->super.a->data, &self->dw->data, &self->dz->data, &batchSize, &self->w->y);
		}
	}
	return self->super.ecx->error;

}

int CamadaFullConnect_learnBatch(CamadaFullConnect self) {
	if (!self->super.params.skipLearn) {
		Execute(kernel_fixW, self->w->length, &self->w->data, &self->dw->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento, &self->w->y);
		Execute(kernel_fixW, self->b->length, &self->b->data, &self->db->data, &self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento, &self->w->y);
	}
	return self->super.ecx->error;
}

char *CamadaFullConnect_json(CamadaFullConnect self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"funcaoAtivacao\":%d", tmp, self->fa);
	gab_free(tmp);

	apendTensor("z", z, string, len, tmp, showValues);
	apendTensor("dz", dz, string, len, tmp, showValues);
	apendTensor("w", w, string, len, tmp, showValues);
	apendTensor("dw", dw, string, len, tmp, showValues);

	apendstr(string, len, "\n}");

	return string;
}

char *CamadaFullConnect_getGenerate(CamadaFullConnect self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s (%zu, %s, Params(%g, %g, %g, %d), RDP(%d, %g, %g), RDP(%d, %g, %g))", lname, self->w->x, F_ATIVACAO_NAME(self->fa), (double) self->super.params.hitlearn, (double) self->super.params.momento, (double) self->super.params.decaimento, self->super.params.skipLearn, self->rdp_pesos.type, self->rdp_pesos.a, self->rdp_pesos.b, self->rdp_bias.type, self->rdp_bias.a, self->rdp_bias.b);
	return string;
}


int CamadaFullConnect_save(CamadaFullConnect self, FILE *f) {
	if (self->super.ecx->error) {
		goto end;
	}
	self->super.ecx->addstack(self->super.ecx, "CamadaFullConnect_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->fa, 1, sizeof(uint32_t), f);
	fwrite(&self->super.s->y, 1, sizeof(size_t), f);
	internal_saveTensor(f, self->w);
	internal_saveTensor(f, self->b);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaFullConnect_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaFullConnect_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;

	uint32_t fa;
	size_t neuronios;

	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&fa, sizeof(uint32_t), 1, f);
	fread(&neuronios, sizeof(size_t), 1, f);
	CamadaFullConnect self = (CamadaFullConnect) CamadaFullConnect_new(gpu, queue, size_in, neuronios, entrada, parametros, fa, ecx, RDP(-1), RDP(-1));
	internal_loadTensor(f, self->w, size_element);
	internal_loadTensor(f, self->b, size_element);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}
int CamadaFullConnect_fprintf(CamadaFullConnect self, FILE * destino, char *format, ...){
	va_list  v;
	va_start(v,format);
	internal_Camada_fprint(self,destino,format,v);
	fprintf(destino,"W -> ");self->w->fprint(self->w,destino);
	fprintf(destino,"dW -> ");self->dw->fprint(self->dw,destino);
	return 0;
}
Camada CamadaFullConnect_new(Gpu gpu, Queue queue, P3d size_in, size_t tamanhoSaida, Tensor entrada, Parametros params, uint32_t funcaoDeAtivacao, Ecx ecx, RandomParams rdp_pesos, RandomParams rdp_bias) {
	ECXPUSH(ecx);
	CamadaFullConnect self = gab_alloc(1, sizeof(CamadaFullConnect_t));
	P3d size_out = {1, tamanhoSaida, 1};
	internal_Camada_new((Camada) self, gpu, queue, FULLCONNECT_ID, lname, params, entrada, size_in, size_out, ecx);
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->b = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->db = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->w = Tensor_new(tamanhoSaida, size_in.x * size_in.y * size_in.z, 1, 1, ecx, 0, gpu->context, queue);
	self->dw = Tensor_new(tamanhoSaida, size_in.x * size_in.y * size_in.z, 1, 1, ecx, 0, gpu->context, queue);
	self->dw->fill(self->dw, 0);
	self->db->fill(self->db, 0);
	self->rdp_pesos = rdp_pesos;
	self->rdp_bias = rdp_bias;
	if (rdp_pesos.type != -1) {
		if (rdp_pesos.type == 0) {
			rdp_pesos = internal_getDefaultRDP(funcaoDeAtivacao == FRELU, size_in.x * size_in.y * size_in.z, self->super.s->length);
		}
		self->super.ecx->error = self->w->randomize(self->w, rdp_pesos.type, rdp_pesos.a, rdp_pesos.b);
		if (ecx->error) {
			goto methods;
		}
	}
	if (rdp_bias.type != -1) {
		if (rdp_bias.type == 0) {
			if (funcaoDeAtivacao == FRELU) {
				self->b->fill(self->b, 0);
			} else {
				rdp_bias = internal_getDefaultRDP(1, size_in.x * size_in.y * size_in.z, self->super.s->length);
				self->super.ecx->error = self->b->randomize(self->b, rdp_bias.type, rdp_bias.a, rdp_bias.b);
			}
		}
		if (ecx->error) {
			goto methods;
		}
	}

	self->fa = funcaoDeAtivacao;
	self->dfa = funcaoDeAtivacao | FLAGDIF;
	//fullfeed(Vector a, Vector w, Vector b, Vector z, Vector s, int fid, int w_x, int w_y, int k0)
	KRN_news(self->fullfeed, "fullfeed", "Vector a, Vector w, Vector b, Vector z, Vector s, int fid, int w_x, int w_y, int k0");
	//fullCalcDWandFix(Vector a, Vector w, Vector dw, Vector dz, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int pesosy, int k0)
	KRN_news(self->fullCalcDWandFix, "fullCalcDWandFix", "Vector a, Vector w, Vector dw, Vector dz, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int pesosy, int k0");
	//fullCalcDzBath(Vector dz, Vector ds, Vector z,  Vector db, int dfa,long batchSize,  int k0)
	KRN_news(self->fullCalcDz, "fullCalcDz", "Vector dz, Vector ds, Vector z,  Vector db, int dfa,long batchSize,  int k0");
	//Vector dz, Vector ds, Vector z, Vector b, Vector db, int dfa, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0
	KRN_news(self->fullCalcDzandFixB, "fullCalcDzAndFixB", "Vector dz, Vector ds, Vector z, Vector b, Vector db, int dfa, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0");
	//fullcalcin(Vector dz, Vector da, Vector w, int pesosx, int pesosy, int k0)
	KRN_news(self->fullcalcin, "fullcalcin", "Vector dz, Vector da, Vector w, int pesosx, int pesosy, int k0");
	//fullCalcDWBatch(Vector a, Vector dw, Vector dz, long batchSize, int pesosy, int k0)
	KRN_news(self->fullCalcDWBatch, "fullCalcDWBatch", "Vector a, Vector dw, Vector dz, long batchSize, int pesosy, int k0");
	//fullCalcDzBath(Vector dz, Vector ds, Vector z, Vector b, Vector db, int dfa,long batchSize,  int k0)
	KRN_news(self->fullCalcDzBath, "fullCalcDzBath", "Vector dz, Vector ds, Vector z, Vector db, int dfa,long batchSize,  int k0");
	//fullFixBatch(Vector w, Vector dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0)
	KRN_news(self->kernel_fixW, "kernel_fixW", "Vector w, Vector dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0");
	ECXPOP(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaFullConnect_release;
	self->super.propagation = (int (*)(void *)) CamadaFullConnect_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaFullConnect_backpropagation;
	self->super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaFullConnect_backpropagationBatch;
	self->super.retroPropagationBatchLearn = (int (*)(void *)) CamadaFullConnect_learnBatch;
	self->super.json = (char *(*)(void *, int)) CamadaFullConnect_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaFullConnect_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaFullConnect_save;
	self->super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaFullConnect_fprintf;
	return (Camada) self;
}

