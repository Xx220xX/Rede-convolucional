//
// Created by hslhe on 14/11/2021.
//

#include "CamadaConv.h"

static const char *lname = "Convolucao";

void CamadaConv_release(CamadaConv *self_p);

int CamadaConv_propagation(CamadaConv self);

int CamadaConv_backpropagation(CamadaConv self, Tensor ds);

char *CamadaConv_json(CamadaConv self, int showValues);

char *CamadaConv_getGenerate(CamadaConv self);

Camada CamadaConv_new(Gpu gpu, Queue queue, Ponto3d passo, Ponto3d filtro, Ponto3d size_in, Tensor entrada,
					  Parametros params, Ecx ecx, RdP rdp_filtros) {
	ecx->addstack(ecx, "CamadaConv_new");
	CamadaConv self = alloc_mem(1, sizeof(CamadaConv_t));
	Ponto3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1,
						filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAO_ID, lname, params, entrada, size_in, size_out, ecx);

	self->grad_filtros = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->filtros = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	if (ecx->error)goto methods;
	self->rdp_filtros = rdp_filtros;
	if (rdp_filtros.type != -1) {
		if (rdp_filtros.type == 0) {
			rdp_filtros.type = TENSOR_UNIFORM;
			rdp_filtros.a = (REAL) (2.0) / (REAL) (size_in.x * size_in.y * size_in.z);
			rdp_filtros.b = -(REAL) 0.5 * rdp_filtros.a;
		}

		self->super.erro->error = self->filtros->randomize(self->filtros, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);
		if (ecx->error)goto methods;

	}

	self->passox = passo.x;
	self->passoy = passo.y;
	self->convSum = Kernel_news(gpu->program, "convSum",
								"Vector filtro, Vector entrada, Vector saida,\n"
								"int passox, int passoy,\n"
								"int saidatx, int saidaty,\n"
								"int entradatx, int entradaty,\n"
								"int fx, int fy, int fz, int k0");
	if (self->super.erro->setError(self->super.erro, self->convSum->error))goto methods;

	self->convCalcGradAndFixWeight = Kernel_news(gpu->program, "convCalcGradAndFixWeight",
												 "Vector filtros, Vector ds, Vector entrada, Vector gradFiltro,\n"
												 "int fx, int fy, int fz,\n"
												 "int entrada_tx, int entrada_ty,\n"
												 "int saida_tx, int saida_ty,\n"
												 "int passox, int passoy,\n"
												 "REAL hitLearn, REAL momento, REAL weightDecay,\n"
												 "int k0");

	if (self->super.erro->setError(self->super.erro, self->convCalcGradAndFixWeight->error))goto methods;

	self->convCalcGradIn = Kernel_news(gpu->program, "convCalcGradIn",
									   "Vector filtro, Vector gradEntrada, Vector gradNext,\n"
									   "int fx, int fy, int fz,\n"
									   "int passox, int passoy,\n"
									   "int entradatx, int entradaty,\n"
									   "int saidatx, int saidaty, int saidatz,\n"
									   "int k0");
	self->super.erro->error = self->convCalcGradIn->error;

	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConv_release;
	self->super.propagation = (int (*)(void *)) CamadaConv_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor *)) CamadaConv_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaConv_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConv_getGenerate;
	return (Camada) self;
}

void CamadaConv_release(CamadaConv *self_p) {
	if (!self_p)return;
	if (!*self_p)return;
	internal_Camada_release((Camada *) self_p);
	(*self_p)->filtros->release(&(*self_p)->filtros);
	(*self_p)->grad_filtros->release(&(*self_p)->grad_filtros);
	if ((*self_p)->convSum)
		(*self_p)->convSum->release(&(*self_p)->convSum);
	if ((*self_p)->convCalcGradIn)
		(*self_p)->convCalcGradIn->release(&(*self_p)->convCalcGradIn);
	if ((*self_p)->convCalcGradAndFixWeight)
		(*self_p)->convCalcGradAndFixWeight->release(&(*self_p)->convCalcGradAndFixWeight);
	free_mem(*self_p);
	*self_p = NULL;
}

int CamadaConv_propagation(CamadaConv self) {
	self->super.erro->setError(self->super.erro,
							   self->convSum->runRecursive(self->convSum, self->super.queue, self->super.s->lenght,
														   *self->super.maxcompute,
														   &self->filtros->data, &self->super.a->data, &self->super.s->data,
														   &self->passox, &self->passoy,
														   &self->super.s->x, &self->super.s->y,
														   &self->super.a->x, &self->super.a->y,
														   &self->filtros->x, &self->filtros->y, &self->filtros->z
							   ));
	return self->super.erro->error;
}

int CamadaConv_backpropagation(CamadaConv self, Tensor ds) {
	if (self->super.da)
		self->super.erro->setError(self->super.erro,
								   self->convCalcGradIn->runRecursive(self->convCalcGradIn, self->super.queue, self->super.da->lenght,
																	  *self->super.maxcompute,
																	  &self->filtros->data, &self->super.da->data, &ds->data,
																	  &self->filtros->x, &self->filtros->y, &self->filtros->z,
																	  &self->passox, &self->passoy,
																	  &self->super.da->x, &self->super.da->y,
																	  &self->super.s->x, &self->super.s->y, &self->super.s->z
								   ));
	if (!self->super.params.disable_learn)
		self->super.erro->setError(self->super.erro,
								   self->convCalcGradAndFixWeight->runRecursive(self->convCalcGradAndFixWeight, self->super.queue,
																				self->filtros->lenght, *self->super.maxcompute,
																				&self->filtros->data, &ds->data,
																				&self->super.a->data, &self->grad_filtros->data,
																				&self->filtros->x, &self->filtros->y, &self->filtros->z,
																				&self->super.a->x, &self->super.a->y,
																				&self->super.s->x, &self->super.s->y,
																				&self->passox, &self->passoy,
																				&self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento
								   ));
	return self->super.erro->error;
}

char *CamadaConv_json(CamadaConv self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = self->filtros->json(self->filtros, showValues);
	apendstr(string, len,
			 "{"
					 PAD"\"passo\":[%zu,%zu],\n"
					 PAD"\"numero_filtros\":%zu,\n"
					 PAD"\"filtros\":%s",
			 self->passox, self->passoy, self->filtros->w,
			 tmp);
	free_mem(tmp);
	tmp = self->grad_filtros->json(self->grad_filtros, showValues);
	apendstr(string, len, ",\n\""PAD"grad_filtros\":%s", tmp);
	free_mem(tmp);
	tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, ",\n"PAD"%s\n}", tmp);
	free_mem(tmp);
	return string;
}

char *CamadaConv_getGenerate(CamadaConv self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "Convolucao(P3D(%zu,%zu),P3D(%zu,%zu,%zu),Params(%g,%g,%g,%d),RDP(%d,%g,%g))", self->passox, self->passoy,
			 self->filtros->x, self->filtros->y, self->filtros->w,
			 (double) self->super.params.hitlearn,
			 (double) self->super.params.momento,
			 (double) self->super.params.decaimento,
			 self->super.params.disable_learn, self->rdp_filtros.type,
			 (double) self->rdp_filtros.a,
			 (double) self->rdp_filtros.b
	)

	return string;
}

//int CamadaConv_save