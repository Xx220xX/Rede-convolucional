//
// Created by hslhe on 19/11/2021.
//

#include <camadas/funcoesDeAtivacao.h>
#include "camadas/CamadaConvNC.h"

static const char *lname = "ConvolucaoNC";

static void CamadaConvNC_release(CamadaConvNC *self_p) {
	if (!self_p)return;
	if (!*self_p)return;
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->filtros);
	Release((*self_p)->grad_filtros);
	Release((*self_p)->z);
	Release((*self_p)->dz);
	Release((*self_p)->convFSum);
	Release((*self_p)->convFCalcGrads);
	Release((*self_p)->convFCalcGradZ);
	Release((*self_p)->convFCalcGradAndFixWeight);
	free_mem(*self_p);
	*self_p = NULL;
}

static int CamadaConvNC_propagation(CamadaConvNC self) {
	Execute(convFSum, self->super.s->lenght,

			&self->filtros->data, &self->super.a->data, &self->z->data, &self->super.s->data,
			&self->passox, &self->passoy,
			&self->super.s->x, &self->super.s->y,
			&self->super.a->x, &self->super.a->y,
			&self->filtros->x, &self->filtros->y, &self->filtros->z,
			&self->activationFuntion
	);
	return self->super.erro->error;
}

static int CamadaConvNC_backpropagation(CamadaConvNC self, Tensor ds) {
	if (self->super.da || !self->super.params.skipLearn) {
		Execute(convFCalcGradZ, self->super.s->lenght,
				&ds->data, &self->z->data,
				&self->dz->data, &self->derivationFuntion
		);
	}
	if (self->super.da)
		Execute(convFCalcGrads, self->super.da->lenght,

				&self->filtros->data, &self->super.da->data, &self->dz->data,
				&self->filtros->x, &self->filtros->y, &self->filtros->z,
				&self->passox, &self->passoy,
				&self->super.da->x, &self->super.da->y,
				&self->super.s->x, &self->super.s->y, &self->super.s->z
		);
	if (!self->super.params.skipLearn)
		Execute(convFCalcGradAndFixWeight,
				self->filtros->lenght,
				&self->filtros->data, &self->dz->data,
				&self->super.a->data, &self->grad_filtros->data,
				&self->filtros->x, &self->filtros->y, &self->filtros->z,
				&self->super.a->x, &self->super.a->y,
				&self->super.s->x, &self->super.s->y,
				&self->passox, &self->passoy,
				&self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento

		);
	return self->super.erro->error;
}

static char *CamadaConvNC_json(CamadaConvNC self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = self->filtros->json(self->filtros, showValues);
	apendstr(string, len,
			 "{"
					 PAD"\"functionActivation\":%d,\n"
					 PAD"\"passo\":[%zu,%zu],\n"
					 PAD"\"numero_filtros\":%zu,\n"
					 PAD"\"filtros\":%s",
			 self->activationFuntion,
			 self->passox, self->passoy, self->filtros->w,
			 tmp);
	free_mem(tmp);
	tmp = self->grad_filtros->json(self->grad_filtros, showValues);
	apendstr(string, len, ",\n"PAD"\"grad_filtros\":%s", tmp);
	free_mem(tmp);
	tmp = self->z->json(self->z, showValues);
	apendstr(string, len, ",\n"PAD"\"z\":%s", tmp);
	free_mem(tmp);
	tmp = self->dz->json(self->dz, showValues);
	apendstr(string, len, ",\n"PAD"\"dz\":%s", tmp);
	free_mem(tmp);
	tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, ",\n"PAD"%s\n}", tmp);
	free_mem(tmp);
	return string;
}

static char *CamadaConvNC_getGenerate(CamadaConvNC self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s(P3D(%zu,%zu),P3D(%zu,%zu,%zu),Params(%g,%g,%g,%d),RDP(%d,%g,%g))",
			 lname,
			 self->passox, self->passoy,
			 self->filtros->x, self->filtros->y, self->filtros->w,
			 (double) self->super.params.hitlearn,
			 (double) self->super.params.momento,
			 (double) self->super.params.decaimento,
			 self->super.params.skipLearn, self->rdp_filtros.type,
			 (double) self->rdp_filtros.a,
			 (double) self->rdp_filtros.b
	)

	return string;
}

/**
 * Salva a camada conv em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 bytes -> separação deve ser '#'
 * 4 bytes -> funcao de ativação
 * 8 bytes -> passo x
 * 8 bytes -> passo y
 * 8 bytes -> filtro x
 * 8 bytes -> filtro y
 * 8 bytes -> filtro z
 * 8 bytes -> filtro w
 * 4 bytes -> filtro size_element
 * size_element*x*y*z*w bytes -> data
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
static int CamadaConvNC_save(CamadaConvNC self, FILE *f) {
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaConvNC_save");
	fwrite(&self->super.layer_id, 1, sizeof(int), f);
	fwrite("#", 1, 1, f);
	fwrite(&self->activationFuntion, 1, sizeof(int), f);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->filtros->x, 1, sizeof(size_t), f);
	fwrite(&self->filtros->y, 1, sizeof(size_t), f);
	fwrite(&self->filtros->z, 1, sizeof(size_t), f);
	fwrite(&self->filtros->w, 1, sizeof(size_t), f);
	fwrite(&self->filtros->size_element, 1, sizeof(unsigned int), f);
	void *data = self->filtros->getvalues(self->filtros, NULL);
	fwrite(data, self->filtros->size_element, self->filtros->lenght, f);
	free_mem(data);

	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

Camada CamadaConvNC_new(Gpu gpu, Queue queue, P2d passo,P2d abertura, P3d filtro, P3d size_in, int ativacao, Tensor entrada,
						Parametros params, Ecx ecx, RdP rdp_filtros) {
	ecx->addstack(ecx, "CamadaConvNC_new");
	CamadaConvNC self = alloc_mem(1, sizeof(CamadaConvNC_t));
	P3d size_out = {(size_in.x - 1 - (filtro.x - 1) * abertura.x) / passo.x + 1,
					(size_in.y - 1 - (filtro.y - 1) * abertura.y) / passo.y + 1,
					filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAO_ID, lname, params, entrada, size_in, size_out, ecx);
	self->activationFuntion = ativacao;
	self->derivationFunction = ativacao | FLAGDIF;
	self->grad_filtros = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->grad_filtros->fill(self->grad_filtros, 0);
	self->filtros = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

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
	self->convFSum = Kernel_news(gpu->program, "convFSum",
								 "Vector filtro, Vector entrada, Vector Z, Vector saida,\n"
								 "int passox, int passoy,\n"
								 "int saidatx, int saidaty,\n"
								 "int entradatx, int entradaty,\n"
								 "int fx, int fy, int fz, int fid, int k0");
	CheckKernel(convFSum);

	self->convFCalcGradZ = Kernel_news(gpu->program, "convFCalcGradZ",
									   "Vector ds, Vector z, Vector dz, int fid, int k0");
	CheckKernel(convFCalcGradZ);

	self->convFCalcGrads = Kernel_news(gpu->program, "convFCalcGradIn",
									   "Vector filtro, Vector gradEntrada, Vector dz,\n"
									   "int fx, int fy, int fz,\n"
									   "int passox, int passoy,\n"
									   "int entradatx, int entradaty,\n"
									   "int saidatx, int saidaty, int saidatz,\n"
									   "int k0");
	CheckKernel(convFCalcGrads);
	self->convFCalcGradAndFixWeight = Kernel_news(gpu->program, "convFCalcGradAndFixWeight",
												  "Vector filtros, Vector dz,\n"
												  "Vector entrada, Vector gradFiltro,\n"
												  "int fx, int fy, int fz,\n"
												  "int entrada_tx, int entrada_ty,\n"
												  "int saida_tx, int saida_ty,\n"
												  "int passox, int passoy,\n"
												  "REAL hitLearn, REAL momento, REAL weightDecay,\n"
												  "int k0");

	CheckKernel(convFCalcGrads);
	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConvNC_release;
	self->super.propagation = (int (*)(void *)) CamadaConvNC_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor )) CamadaConvNC_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaConvNC_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConvNC_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaConvNC_save;
	return (Camada) self;
}


