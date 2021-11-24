//
// Created by hslhe on 14/11/2021.
//

#include "camadas/CamadaConv.h"

static const char *lname = "Convolucao";

static void CamadaConv_release(CamadaConv *self_p) {
	if (!self_p)return;
	if (!*self_p)return;
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->filtros);
	Release((*self_p)->grad_filtros);
	Release((*self_p)->convSum);
	Release((*self_p)->convCalcGradIn);
	Release((*self_p)->convCalcGradAndFixWeight);
	free_mem(*self_p);
	*self_p = NULL;
}

static int CamadaConv_propagation(CamadaConv self) {
	Execute(convSum, self->super.s->length,

			&self->filtros->data, &self->super.a->data, &self->super.s->data,
			&self->passox, &self->passoy,
			&self->super.s->x, &self->super.s->y,
			&self->super.a->x, &self->super.a->y,
			&self->filtros->x, &self->filtros->y, &self->filtros->z
	);
	return self->super.erro->error;
}

static int CamadaConv_backpropagation(CamadaConv self, Tensor ds) {
	if (self->super.da)
		Execute(convCalcGradIn, self->super.da->length,

				&self->filtros->data, &self->super.da->data, &ds->data,
				&self->filtros->x, &self->filtros->y, &self->filtros->z,
				&self->passox, &self->passoy,
				&self->super.da->x, &self->super.da->y,
				&self->super.s->x, &self->super.s->y, &self->super.s->z
		);
	if (!self->super.params.skipLearn)
		Execute(convCalcGradAndFixWeight,
				self->filtros->length,
				&self->filtros->data, &ds->data,
				&self->super.a->data, &self->grad_filtros->data,
				&self->filtros->x, &self->filtros->y, &self->filtros->z,
				&self->super.a->x, &self->super.a->y,
				&self->super.s->x, &self->super.s->y,
				&self->passox, &self->passoy,
				&self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento
		);
	return self->super.erro->error;
}

static char *CamadaConv_json(CamadaConv self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{\n"
					 PAD"%s,\n"
					 PAD"\"passo\":[%zu,%zu],\n"
					 PAD"\"numero_filtros\":%zu",
			 tmp, self->passox, self->passoy, self->filtros->w
	);
	free_mem(tmp);
	apendTensor("filtros", filtros, string, len, tmp, showValues);
	apendTensor("grad_filtros", grad_filtros, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	return string;
}

static char *CamadaConv_getGenerate(CamadaConv self) {
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
	);

	return string;
}

/**
 * Salva a camada conv em um arquivo
 * 4 bytes -> indicando o tipo
 * 4 bytes -> separação deve ser -1
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
static int CamadaConv_save(CamadaConv self, FILE *f) {
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaConv_save");
	int flag = -1;
	fwrite(&self->super.layer_id, 1, sizeof(int), f);
	fwrite(&flag, 1, sizeof(int), f);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->filtros->x, 1, sizeof(size_t), f);
	fwrite(&self->filtros->y, 1, sizeof(size_t), f);
	fwrite(&self->filtros->z, 1, sizeof(size_t), f);
	fwrite(&self->filtros->w, 1, sizeof(size_t), f);
	fwrite(&self->filtros->size_element, 1, sizeof(unsigned int), f);

	fwrite(&self->super.params.hitlearn, 1, sizeof(REAL), f);
	fwrite(&self->super.params.momento, 1, sizeof(REAL), f);
	fwrite(&self->super.params.decaimento, 1, sizeof(REAL), f);
	fwrite(&self->super.params.skipLearn, 1, sizeof(int), f);
	void *data = self->filtros->getvalues(self->filtros, NULL);
	fwrite(data, self->filtros->size_element, self->filtros->length, f);
	free_mem(data);

	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

Camada CamadaConv_load(FILE *f, Gpu gpu, Queue queue, P3d sizein, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaConv_save");
	int flag;
	fread(&flag, 1, sizeof(int), f);
	if (flag != -1) {
		fread(&flag, 1, sizeof(int), f);
	}
	union {
		double auxd;
		float auxf;
		REAL auxR;
	} aux;
	P2d passo;
	P3d filtros;
	Parametros prm;
	size_t nfiltros, length;
	unsigned int size_element;
	fread(&passo.x, 1, sizeof(size_t), f);
	fread(&passo.y, 1, sizeof(size_t), f);
	fread(&filtros.x, 1, sizeof(size_t), f);
	fread(&filtros.y, 1, sizeof(size_t), f);
	fread(&filtros.z, 1, sizeof(size_t), f);
	fread(&nfiltros, 1, sizeof(size_t), f);
	fread(&size_element, 1, sizeof(unsigned int), f);

	fread(&aux, 1, size_element, f);
	REALCAST(prm.hitlearn, size_element, aux);
	fread(&aux, 1, size_element, f);
	REALCAST(prm.momento, size_element, aux);
	fread(&aux, 1, size_element, f);
	REALCAST(prm.decaimento, size_element, aux);
	fread(&prm.skipLearn, 1, sizeof(int), f);
	length = filtros.x * filtros.y * filtros.z * nfiltros;

	void *data = alloc_mem(length, size_element);
	fread(data, size_element, length, f);

	CamadaConv self = CamadaConv_new(gpu, queue, passo, filtros, sizein, entrada, prm, ecx, Rdp(-1));
	if (size_element != sizeof(REAL)) {
		REAL *dt = alloc_mem(length, sizeof(REAL));
		if (size_element == sizeof(double)) {
			for (int i = 0; i < length; ++i)
				dt[i] = (REAL) ((double *) data)[i];
		} else {
			for (int i = 0; i < length; ++i)
				dt[i] = (REAL) ((float *) data)[i];
		}
		free_mem(data);
		data = dt;

	}
	self->filtros->setvalues(self->filtros, data);
	free_mem(data);

	end:
	ecx->popstack(ecx);
	return (Camada) self;
}

Camada CamadaConv_new(Gpu gpu, Queue queue, P2d passo, P3d filtro, P3d size_in, Tensor entrada,
					  Parametros params, Ecx ecx, RandomParams rdp_filtros) {
	ecx->addstack(ecx, "CamadaConv_new");
	CamadaConv self = alloc_mem(1, sizeof(CamadaConv_t));
	P3d size_out = {(size_in.x - filtro.x) / passo.x + 1, (size_in.y - filtro.y) / passo.y + 1,
					filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAO_ID, lname, params, entrada, size_in, size_out, ecx);

	self->grad_filtros = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->grad_filtros->fill(self->grad_filtros, 0);
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
	CheckKernel(convSum);

	self->convCalcGradAndFixWeight = Kernel_news(gpu->program, "convCalcGradAndFixWeight",
												 "Vector filtros, Vector ds, Vector entrada, Vector gradFiltro,\n"
												 "int fx, int fy, int fz,\n"
												 "int entrada_tx, int entrada_ty,\n"
												 "int saida_tx, int saida_ty,\n"
												 "int passox, int passoy,\n"
												 "REAL hitLearn, REAL momento, REAL weightDecay,\n"
												 "int k0");

	CheckKernel(convCalcGradAndFixWeight);

	self->convCalcGradIn = Kernel_news(gpu->program, "convCalcGradIn",
									   "Vector filtro, Vector gradEntrada, Vector gradNext,\n"
									   "int fx, int fy, int fz,\n"
									   "int passox, int passoy,\n"
									   "int entradatx, int entradaty,\n"
									   "int saidatx, int saidaty, int saidatz,\n"
									   "int k0");
	CheckKernel(convCalcGradIn);
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
