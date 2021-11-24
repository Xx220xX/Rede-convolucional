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
	Release((*self_p)->W);
	Release((*self_p)->dW);
	Release((*self_p)->z);
	Release((*self_p)->dz);
	Release((*self_p)->convncSum);
	Release((*self_p)->convncCalcFiltro);
	Release((*self_p)->convncCalcGradZ);
	Release((*self_p)->convncCalcGrads);
	free_mem(*self_p);
	*self_p = NULL;
}

static int CamadaConvNC_propagation(CamadaConvNC self) {
	Execute(convncSum, self->super.s->length,
			&self->W->data, &self->super.a->data, &self->z->data, &self->super.s->data,
			&self->passox, &self->passoy,
			&self->activationFuntion,
			&self->passox, &self->passoy,
			&self->largx, &self->largy,
			&self->super.a->x, &self->super.a->y,
			&self->super.s->x, &self->super.s->y,
			&self->W->x, &self->W->y, &self->W->z
	);
	return self->super.erro->error;
}

static int CamadaConvNC_backpropagation(CamadaConvNC self, Tensor ds) {
	if (self->super.da || !self->super.params.skipLearn) {
		Execute(convncSum, self->super.s->length,
				&ds->data, &self->z->data,
				&self->dz->data, &self->derivationFunction
		);
	}
	if (self->super.da) {
		Execute(convncCalcGrads, self->super.da->length,

				&self->W->data, &self->super.da->data, &self->dz->data,
				&self->passox, &self->passoy,
				&self->super.da->x, &self->super.da->y,
				&self->super.s->x, &self->super.s->y, &self->super.s->z,
				&self->W->x, &self->W->y, &self->W->z
		);
	}
	if (!self->super.params.skipLearn)
		Execute(convncCalcFiltro,
				self->W->length,
				&self->dz->data, &self->super.a->data,
				&self->W->data, &self->dW->data,
				&self->dW->x, &self->dW->y, &self->dW->z,
				&self->super.a->x, &self->super.a->y,
				&self->super.s->x, &self->super.s->y,
				&self->passox, &self->passoy,
				&self->largx, &self->largy,
				&self->super.params.hitlearn, &self->super.params.momento, &self->super.params.decaimento

		);
	return self->super.erro->error;
}

static char *CamadaConvNC_json(CamadaConvNC self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{"
					 PAD"%s,\n"
					 PAD"\"functionActivation\":%d,\n"
					 PAD"\"passo\":[%zu,%zu],\n"
					 PAD"\"largura\":[%zu,%zu],\n"
					 PAD"\"numero_filtros\":%zu",
			 tmp,
			 self->activationFuntion,
			 self->passox, self->passoy,
			 self->largx, self->largy,
			 self->W->w
			 );

	apendTensor("W",W,string,len,tmp,showValues);
	apendTensor("dW",dW,string,len,tmp,showValues);
	apendTensor("Z",z,string,len,tmp,showValues);
	apendTensor("dZ",dz,string,len,tmp,showValues);

	apendstr(string, len, "\n}");

	return string;
}

static char *CamadaConvNC_getGenerate(CamadaConvNC self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s(P2D(%zu,%zu),P2D(%zu,%zu),P3D(%zu,%zu,%zu),Params(%g,%g,%g,%d),RDP(%d,%g,%g))",
			 lname,
			 self->passox, self->passoy,
			 self->largx, self->largy,
			 self->W->x, self->W->y, self->W->w,
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
 * 1 bytes -> separação deve ser '#'
 * 4 bytes -> funcao de ativação
 * 8 bytes -> passo x
 * 8 bytes -> passo y
 * 8 bytes -> largura x
 * 8 bytes -> largura y
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
	fwrite(&self->largx, 1, sizeof(size_t), f);
	fwrite(&self->largy, 1, sizeof(size_t), f);
	fwrite(&self->W->x, 1, sizeof(size_t), f);
	fwrite(&self->W->y, 1, sizeof(size_t), f);
	fwrite(&self->W->z, 1, sizeof(size_t), f);
	fwrite(&self->W->w, 1, sizeof(size_t), f);
	fwrite(&self->W->size_element, 1, sizeof(unsigned int), f);
	void *data = self->W->getvalues(self->W, NULL);
	fwrite(data, self->W->size_element, self->W->length, f);
	free_mem(data);
	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

Camada CamadaConvNC_new(Gpu gpu, Queue queue, P2d passo, P2d abertura, P3d filtro, P3d size_in,
						uint32_t ativacao, Tensor entrada,
						Parametros params, Ecx ecx, RandomParams rdp_filtros) {
	ecx->addstack(ecx, "CamadaConvNC_new");
	CamadaConvNC self = alloc_mem(1, sizeof(CamadaConvNC_t));
	P3d size_out = {(size_in.x - 1 - (filtro.x - 1) * abertura.x) / passo.x + 1,
					(size_in.y - 1 - (filtro.y - 1) * abertura.y) / passo.y + 1,
					filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAONC_ID, lname, params, entrada, size_in, size_out, ecx);
	self->activationFuntion = ativacao;
	self->derivationFunction = ativacao | FLAGDIF;
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
			rdp_filtros.a = (REAL) (2.0) / (REAL) (size_in.x * size_in.y * size_in.z);
			rdp_filtros.b = -(REAL) 0.5 * rdp_filtros.a;
		}
		self->super.erro->error = self->W->randomize(self->W, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);
		if (ecx->error)goto methods;

	}

	self->passox = passo.x;
	self->passoy = passo.y;
	self->convncSum = Kernel_news(gpu->program, "convncSum",
								  "Vector W, Vector A, Vector Z, Vector S,\n"
								  "unsigned int fid,\n"
								  "unsigned int passox, int passoy,\n"
								  "unsigned int largx, unsigned int largy,\n"
								  "unsigned int entradatx, unsigned int entradaty,\n"
								  "unsigned int saidatx, unsigned int saidaty,\n"
								  "unsigned int fx, unsigned int fy, unsigned int fz,\n"
								  "int k0");
	CheckKernel(convncSum);

	self->convncCalcGradZ = Kernel_news(gpu->program, "convncCalcGradZ",
										"Vector ds, Vector z, Vector dz, unsigned int fid, int k0");
	CheckKernel(convncCalcGradZ);

	self->convncCalcFiltro = Kernel_news(gpu->program, "convncCalcFiltro",
										 "Vector dz,\n"
										 "Vector A,\n"
										 "Vector W,\n"
										 "Vector dW,\n"
										 "unsigned int dw_x, unsigned int dw_y, unsigned int dw_z,\n"
										 "unsigned int a_x, unsigned int a_y,\n"
										 "unsigned int s_x, unsigned int s_y,\n"
										 "unsigned int passox, unsigned int passoy,\n"
										 "unsigned int largx, unsigned int largy,\n"
										 "REAL hitlearn, REAL momento, REAL weightDecay,\n"
										 "int k0");
	CheckKernel(convncCalcFiltro);
	self->convncCalcGrads = Kernel_news(gpu->program, "convncCalcGrads",
										"Vector W,\n"
										"Vector DA,\n"
										"Vector dz,\n"
										"unsigned int passox, unsigned int passoy,\n"
										"unsigned int largx, unsigned int largy,\n"
										"unsigned int entradatx, unsigned int entradaty,\n"
										"unsigned int saidatx, unsigned int saidaty,\n"
										"unsigned int fx, unsigned int fy, unsigned int fz,\n"
										"int k0");

	CheckKernel(convncCalcGrads);
	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaConvNC_release;
	self->super.propagation = (int (*)(void *)) CamadaConvNC_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaConvNC_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaConvNC_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaConvNC_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaConvNC_save;
	return (Camada) self;
}


