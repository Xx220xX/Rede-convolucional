//
// Created by hslhe on 19/11/2021.
//


#include "camadas/CamadaConvF.h"

static const char *lname = "ConvolucaoF";

static void CamadaConvF_release(CamadaConvF *self_p) {
	if (!self_p)return;
	if (!*self_p)return;
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->W);
	Release((*self_p)->dW);
	Release((*self_p)->z);
	Release((*self_p)->dz);
	Release((*self_p)->convFSum);
	Release((*self_p)->convFCalcGrads);
	Release((*self_p)->convFCalcGradZ);
	Release((*self_p)->convFCalcGradAndFixWeight);
	free_mem(*self_p);
	*self_p = NULL;
}

static int CamadaConvF_propagation(CamadaConvF self) {
	Execute(convFSum, self->super.s->length,

			&self->W->data, &self->super.a->data, &self->z->data, &self->super.s->data,
			&self->passox, &self->passoy,
			&self->super.s->x, &self->super.s->y,
			&self->super.a->x, &self->super.a->y,
			&self->W->x, &self->W->y, &self->W->z,
			&self->activationFuntion
	);
	return self->super.erro->error;
}

static int CamadaConvF_backpropagation(CamadaConvF self, Tensor ds) {
	if (self->super.da || !self->super.params.skipLearn) {
		Execute(convFCalcGradZ, self->super.s->length,
				&ds->data, &self->z->data,
				&self->dz->data, &self->derivationFuntion
		);
	}
	if (self->super.da)
		Execute(convFCalcGrads, self->super.da->length,

				&self->W->data, &self->super.da->data, &self->dz->data,
				&self->W->x, &self->W->y, &self->W->z,
				&self->passox, &self->passoy,
				&self->super.da->x, &self->super.da->y,
				&self->super.s->x, &self->super.s->y, &self->super.s->z
		);
	if (!self->super.params.skipLearn)
		Execute(convFCalcGradAndFixWeight,
				self->W->length,
				&self->W->data, &self->dz->data,
				&self->super.a->data, &self->dW->data,
				&self->W->x, &self->W->y, &self->W->z,
				&self->super.a->x, &self->super.a->y,
				&self->super.s->x, &self->super.s->y,
				&self->passox, &self->passoy,
				&self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento

		);
	return self->super.erro->error;
}

static char *CamadaConvF_json(CamadaConvF self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{"
					 PAD"%s,\n"
					 PAD"\"functionActivation\":%d,\n"
					 PAD"\"passo\":[%zu,%zu],\n"
					 PAD"\"numero_filtros\":%zu", tmp,
			 self->activationFuntion,
			 self->passox, self->passoy, self->W->w);
	free_mem(tmp);
	apendTensor("filtros", W, string, len, tmp, showValues);
	apendTensor("grad_filtros", dW, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaConvF_getGenerate(CamadaConvF self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s (P2D(%zu, %zu), P3D(%zu, %zu, %zu), %d, Params(%g, %g, %g, %d), RDP(%d, %g, %g))",
			 lname,
			 self->passox, self->passoy,
			 self->W->x, self->W->y, self->W->w,
			 self->activationFuntion,
			 (double) self->super.params.hitlearn,
			 (double) self->super.params.momento,
			 (double) self->super.params.decaimento,
			 self->super.params.skipLearn, self->rdp_filtros.type,
			 (double) self->rdp_filtros.a,
			 (double) self->rdp_filtros.b
	);

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
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaConvF_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->W->x, 1, sizeof(size_t), f);
	fwrite(&self->W->y, 1, sizeof(size_t), f);
	fwrite(&self->W->w, 1, sizeof(size_t), f);
	fwrite(&self->activationFuntion, 1, sizeof(uint32_t), f);
	internal_saveTensor(f, self->W);
	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
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

Camada CamadaConvF_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, uint32_t ativacao, Tensor entrada, Parametros params, Ecx ecx, RandomParams rdp_filtros) {
	ECXPOP(ecx);
	CamadaConvF self = alloc_mem(1, sizeof(CamadaConvF_t));
	P3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1, filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAOF_ID, lname, params, entrada, size_in, size_out, ecx);
	self->activationFuntion = ativacao;
	self->derivationFuntion = ativacao | FLAGDIF;
	self->dW = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->dW->fill(self->dW, 0);
	self->W = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	if (ecx->error)goto methods;
	self->rdp_filtros = rdp_filtros;
	if (rdp_filtros.type != -1) {
		if (rdp_filtros.type == 0) {
			rdp_filtros.type = TENSOR_UNIFORM;
			rdp_filtros.a = (REAL) (2.0) / (REAL) (filtro.x * filtro.y * size_in.z);
			rdp_filtros.b = -(REAL) 0.5 * rdp_filtros.a;
		}

		self->super.erro->error = self->W->randomize(self->W, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);
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
	ECXPOP(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConvF_release;
	self->super.propagation = (int (*)(void *)) CamadaConvF_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaConvF_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaConvF_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConvF_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaConvF_save;
	return (Camada) self;
}


