//
// Created by Henrique on 21/11/2021.
//

#include <math.h>
#include "cnn.h"

#include"camadas/CamadaConv.h"
#include"camadas/CamadaConvF.h"
#include"camadas/CamadaConvNC.h"
#include"camadas/CamadaPool.h"
#include"camadas/CamadaRelu.h"
#include"camadas/CamadaPRelu.h"
#include"camadas/CamadaFullConnect.h"
#include"camadas/CamadaPadding.h"
#include"camadas/CamadaDropOut.h"
#include"camadas/CamadaSoftMax.h"
#include"camadas/CamadaBatchNorm.h"
#include "kernel_lib.h"


#define NKERNELS 3
#define CNN_KERNEL_SUB 0
#define CNN_KERNEL_NORMALIZE_CHAR_2_REAL 1
#define CNN_KERNEL_EXTRACT_VECTOR_CLASS_FROM_CHAR 2


#define CHECKDIN(input, filtro, abertura, passo) \
    (((((input-1) - (filtro - 1) * abertura) / passo +1)>0) && \
    (((((input-1) - (filtro - 1) * abertura) / passo)*passo + (filtro-1)*abertura) == (input-1)))

int Cnn_release(Cnn *selfp);

void internal_Cnn_getKernels(Cnn self);

int internal_Cnn_addlayer(Cnn self, Camada newLayer);

Tensor internal_Cnn_getEntrada(Cnn self);


P3d Cnn_getSizeOut(Cnn self);

void Cnn_removeLastLayer(Cnn self);

int Cnn_predict(Cnn self, Tensor entrada);

int Cnn_predictv(Cnn self, REAL *entrada);

int Cnn_learn(Cnn self, Tensor target);

int Cnn_learnv(Cnn self, REAL *target);

REAL Cnn_mse(Cnn self);

REAL Cnn_mseT(Cnn self, Tensor target);

int Cnn_normalizeIMAGE(Cnn self, Tensor dst, Tensor src);

int Cnn_extractVectorLabelClass(Cnn self, Tensor dst, Tensor label);

char *Cnn_json(Cnn self, int showValues);
void Cnn_jsonF(Cnn self, int showValues,const char *filename);

int Cnn_setInput(Cnn self, size_t x, size_t y, size_t z);

//#####################################################################################
//							FUNÇÕES PARA ADICIONA CAMADAS
//#####################################################################################
int Cnn_Convolucao(Cnn self, P2d passo, P3d filtro, Parametros p, RandomParams filtros);

int Cnn_ConvolucaoF(Cnn self, P2d passo, P3d filtro, uint32_t funcaoAtivacao, Parametros p, RandomParams filtros);

int Cnn_ConvolucaoNC(Cnn self, P2d passo, P2d abertura, P3d filtro, uint32_t funcaoAtivacao, Parametros p, RandomParams filtros);

int Cnn_Pooling(Cnn self, P2d passo, P2d filtro, uint32_t type);

int Cnn_Relu(Cnn self, REAL fator_menor0, REAL fator_maior0);

int Cnn_PRelu(Cnn self, Parametros params, RandomParams rdp_a);

int Cnn_FullConnect(Cnn self, size_t numero_neuronios, Parametros p, uint32_t funcaoAtivacao, RandomParams rdp_pesos, RandomParams rdp_bias);

int Cnn_Padding(Cnn self, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);

int Cnn_DropOut(Cnn self, REAL probabilidadeSaida, cl_ulong seed);

int Cnn_SoftMax(Cnn self);

int Cnn_BatchNorm(Cnn self, REAL epsilon, Parametros p, RandomParams randY, RandomParams randB);



//#####################################################################################
//
//									CORPO DAS FUNÇÕES
//
//#####################################################################################





Cnn Cnn_new(Gpu gpu) {
	Cnn self = alloc_mem(1, sizeof(Cnn_t));
	self->erro = Ecx_new(0);

	if (gpu) {
		self->gpu = gpu;
	} else {
		self->gpu = Gpu_new();
		self->erro->error = self->gpu->compileProgram(self->gpu, KERNEL_LIB_get_defaultKernel());
		self->release_gpu = 1;
	}
	self->queue = self->gpu->Queue_new(self->gpu, self->erro->perro);
	internal_Cnn_getKernels(self);
	methods:
	self->Convolucao = Cnn_Convolucao;
	self->ConvolucaoF = Cnn_ConvolucaoF;
	self->ConvolucaoNC = Cnn_ConvolucaoNC;
	self->Pooling = Cnn_Pooling;
	self->Relu = Cnn_Relu;
	self->PRelu = Cnn_PRelu;
	self->FullConnect = Cnn_FullConnect;
	self->Padding = Cnn_Padding;
	self->DropOut = Cnn_DropOut;
	self->SoftMax = Cnn_SoftMax;
	self->BatchNorm = Cnn_BatchNorm;

	self->predict = Cnn_predict;
	self->predictv = Cnn_predictv;
	self->learn = Cnn_learn;
	self->learnv = Cnn_learnv;
	self->mse = Cnn_mse;
	self->mseT = Cnn_mseT;
	self->normalizeIMAGE = Cnn_normalizeIMAGE;
	self->extractVectorLabelClass = Cnn_extractVectorLabelClass;
	self->json = Cnn_json;
	self->jsonF = Cnn_jsonF;
	self->setInput = Cnn_setInput;

	self->removeLastLayer = Cnn_removeLastLayer;
	self->getSizeOut = Cnn_getSizeOut;
	self->release = Cnn_release;

	return self;
}

int Cnn_release(Cnn *selfp) {
	if (!selfp)return 10;
	if (!(*selfp))return 10;
	int erro = (*selfp)->erro->error;
	for (int l = 0; l < (*selfp)->l; ++l) {
		(*selfp)->cm[l]->release(&(*selfp)->cm[l]);
	}
	if ((*selfp)->cm)free_mem((*selfp)->cm);

	Release((*selfp)->entrada);
	Release((*selfp)->target);
	Release((*selfp)->ds);
	Kernel *kernels = (*selfp)->kernels;
	if ((*selfp)->kernels) {
		for (int i = 0; i < NKERNELS; ++i) {
			Release(kernels[i]);
		}

		free_mem((*selfp)->kernels);
	}
	if ((*selfp)->release_gpu) {
		Release((*selfp)->gpu);
	}
	free_mem(*selfp);
	return erro;
}

void internal_Cnn_getKernels(Cnn self) {
	Kernel sub = Kernel_news(self->gpu->program, "kernel_sub", "Vector ds, Vector s, Vector t, int k0");
	Kernel normalizechar2real = Kernel_news(self->gpu->program, "kernel_normalizechar2real", "Vector dst, __global char *src, REAL a, REAL b, int k0");
	self->erro->setError(self->erro, normalizechar2real->error);
	Kernel getVetorClassFromChar = Kernel_news(self->gpu->program, "kernel_getVetorClassFromChar", "__global unsigned char *ints, Vector v, int noptiobs, int k0");
	self->erro->setError(self->erro, normalizechar2real->error);
	Kernel *allkernels = alloc_mem(NKERNELS, sizeof(Kernel));
	self->erro->setError(self->erro, normalizechar2real->error);

	allkernels[CNN_KERNEL_SUB] = sub;
	allkernels[CNN_KERNEL_NORMALIZE_CHAR_2_REAL] = normalizechar2real;
	allkernels[CNN_KERNEL_EXTRACT_VECTOR_CLASS_FROM_CHAR] = getVetorClassFromChar;
	self->kernels = allkernels;
}

int internal_Cnn_addlayer(Cnn self, Camada newLayer) {
	if (self->erro->error) {
		newLayer->release(newLayer);
		return self->erro->error;
	}
	self->l = self->l + 1;
	self->cm = realloc(self->cm, self->l * sizeof(Camada));
	self->cm[self->l - 1] = newLayer;
	Release(self->target);
	Release(self->ds);
	P3d size_out = newLayer->getOutSize(newLayer);
	self->target = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->erro, 0, self->gpu->context, self->queue);
	self->ds = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->erro, 0, self->gpu->context, self->queue);
}

Tensor internal_Cnn_getEntrada(Cnn self) {
	if (self->cm)return self->cm[self->l - 1]->s;
	return NULL;
}

P3d Cnn_getSizeOut(Cnn self) {
	if (self->l > 0) {
		return self->cm[self->l - 1]->getOutSize(self->cm[self->l - 1]);
	}
	return self->size_in;
}

void Cnn_removeLastLayer(Cnn self) {
	if (self->l <= 0)return;
	self->l = self->l - 1;
	Release(self->cm[self->l]);
	if (self->l == 0) {
		free_mem(self->cm);
		self->cm = NULL;
	} else
		self->cm = realloc_mem(self->cm, self->l * sizeof(Camada));

	Release(self->target);
	Release(self->ds);
	P3d size_out = self->getSizeOut(self);
	self->target = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->erro, 0, self->gpu->context, self->queue);
	self->ds = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->erro, 0, self->gpu->context, self->queue);
}

int Cnn_predict(Cnn self, Tensor entrada) {
	if (!self->l)self->erro->error = 34;
	if (self->erro->error) return self->erro->error;
	if (entrada->flag.ram || entrada->flag.shared) {
		return Cnn_predictv(self, entrada->data);
	}
	self->cm[0]->a = entrada;
	for (int l = 0; l < self->l && !self->erro->error; ++l) {
		self->cm[l]->propagation(self->cm[l]);
	}
	return self->erro->error;
}

int Cnn_predictv(Cnn self, REAL *entrada) {
	if (!entrada) self->erro->error = 33;
	if (!self->l)self->erro->error = 34;
	if (self->erro->error) return self->erro->error;
	self->entrada->setvalues(self->entrada, entrada);
	return Cnn_predict(self, self->entrada);

}


int Cnn_learn(Cnn self, Tensor target) {
	if (!self->l)self->erro->error = 34;
	if (self->erro->error) return self->erro->error;
	if (target->flag.ram || target->flag.shared) {
		return Cnn_learnv(self, target->data);
	}
	Kernel sub = ((Kernel *) self->kernels)[CNN_KERNEL_SUB];
	sub->runRecursive(sub, self->queue, self->ds->length, self->gpu->maxworks, &self->ds->data,
					  &self->cm[self->l - 1]->s->data, &target->data);
	Tensor ds = self->ds;
	for (int l = 0; l < self->l && !self->erro->error; ++l) {
		self->cm[l]->retroPropagation(self->cm[l], ds);
		ds = self->cm[l]->da;
	}
	return self->erro->error;
}

int Cnn_learnv(Cnn self, REAL *target) {
	if (!target) self->erro->error = 33;
	if (!self->l)self->erro->error = 34;
	if (self->erro->error) return self->erro->error;
	self->target->setvalues(self->entrada, target);
	return Cnn_learn(self, self->target);
}

REAL Cnn_mse(Cnn self) {
	if (self->l <= 0)return NAN;
	REAL mse = 0;
	REAL *data = self->ds->getvalues(self->ds, NULL);
	for (int i = 0; i < self->ds->length; ++i) {
		mse += data[i] * data[i];
	}
	free_mem(data);
	return mse / 2;

}

REAL Cnn_mseT(Cnn self, Tensor target) {
	if (self->l <= 0)return NAN;
	REAL *data = self->cm[self->l - 1]->s->getvalues(self->ds, NULL);
	REAL *dataT = self->cm[self->l - 1]->s->getvalues(self->ds, NULL);
	REAL mse = 0;
	REAL tmp;
	for (int i = 0; i < self->cm[self->l - 1]->s->length; ++i) {
		tmp = data[i] - dataT[i];
		mse += tmp * tmp;
	}
	free_mem(data);
	free_mem(dataT);
	return mse / 2;
}
//#####################################################################################
//
//						CORPO DAS FUNÇÕES DE ADICIONAR CAMADAS
//
//#####################################################################################



int Cnn_Convolucao(Cnn self, P2d passo, P3d filtro, Parametros p, RandomParams filtros) {
	P3d size_in = self->getSizeOut(self);
	if (!CHECKDIN(size_in.x, filtro.x, 1, passo.x) ||
		!CHECKDIN(size_in.y, filtro.y, 1, passo.y)) {
		fprintf(stderr, "Convolucao:Invalid params\nsize in : %zu %zu %zu\nsize out : %g %g %zu\n",
				size_in.x, size_in.y, size_in.z,
				(size_in.x - 1 - (filtro.x - 1)) / (REAL)passo.x + 1,
				(size_in.y - 1 - (filtro.y - 1)) / (REAL)passo.y + 1,
				size_in.z
		);
		return 25;
	}
	Camada c = CamadaConv_new(self->gpu, self->queue, passo, filtro, size_in, internal_Cnn_getEntrada(self),
							  p, self->erro, filtros);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_ConvolucaoF(Cnn self, P2d passo, P3d filtro, uint32_t funcaoAtivacao, Parametros p, RandomParams filtros) {
	P3d size_in = self->getSizeOut(self);
	if (!CHECKDIN(size_in.x, filtro.x, 1, passo.x) ||
		!CHECKDIN(size_in.y, filtro.y, 1, passo.y)) {
		fprintf(stderr, "ConvolucaoF:Invalid params\nsize in : %zu %zu %zu\nsize out : %g %g %zu\n",
				size_in.x, size_in.y, size_in.z,
				(size_in.x - 1 - (filtro.x - 1)) /(REAL) passo.x + 1,
				(size_in.y - 1 - (filtro.y - 1)) /(REAL) passo.y + 1,
				size_in.z
		);
		return 25;
	}
	Camada c = CamadaConvF_new(self->gpu, self->queue, passo, filtro, size_in, funcaoAtivacao, internal_Cnn_getEntrada(self), p, self->erro, filtros);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_ConvolucaoNC(Cnn self, P2d passo, P2d abertura, P3d filtro, uint32_t funcaoAtivacao, Parametros p, RandomParams filtros) {
	P3d size_in = self->getSizeOut(self);
	if (!CHECKDIN(size_in.x, filtro.x, abertura.x, passo.x) ||
		!CHECKDIN(size_in.y, filtro.y, abertura.y, passo.y)) {
		fprintf(stderr, "ConvolucaoNC:Invalid params\nsize in : %zu %zu %zu\nsize out : %zu %zu %zu\n",
				size_in.x, size_in.y, size_in.z,
				(size_in.x - 1 - (filtro.x - 1) * abertura.x) / passo.x + 1,
				(size_in.y - 1 - (filtro.y - 1) * abertura.y) / passo.y + 1,
				size_in.z
		);
		return 25;
	}
	Camada c = CamadaConvNC_new(self->gpu, self->queue, passo, abertura, filtro, size_in, funcaoAtivacao,
								internal_Cnn_getEntrada(self), p, self->erro, filtros);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_Pooling(Cnn self, P2d passo, P2d filtro, uint32_t type) {
	P3d size_in = self->getSizeOut(self);
	if (!CHECKDIN(size_in.x, filtro.x, 1, passo.x) ||
		!CHECKDIN(size_in.y, filtro.y, 1, passo.y)) {
		fprintf(stderr, "Pooling:Invalid params\nsize in : %zu %zu %zu\nsize out : %g %g %zu\n",
				size_in.x, size_in.y, size_in.z,
				(size_in.x - 1 - (filtro.x - 1)) /(REAL) passo.x + 1,
				(size_in.y - 1 - (filtro.y - 1)) /(REAL) passo.y + 1,
				size_in.z
		);
		return 25;
	}
	Camada c = CamadaPool_new(self->gpu, self->queue, passo, filtro, size_in, type, internal_Cnn_getEntrada(self), self->erro);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_Relu(Cnn self, REAL fator_menor0, REAL fator_maior0) {
	Camada c = CamadaRelu_new(self->gpu, self->queue, self->getSizeOut(self), fator_menor0, fator_maior0, internal_Cnn_getEntrada(self), self->erro);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_PRelu(Cnn self, Parametros params, RandomParams rdp_a) {
	Camada c = CamadaPRelu_new(self->gpu, self->queue, self->getSizeOut(self), internal_Cnn_getEntrada(self), params, rdp_a, self->erro);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_FullConnect(Cnn self, size_t numero_neuronios, Parametros p, uint32_t funcaoAtivacao, RandomParams rdp_pesos, RandomParams rdp_bias) {
	Camada c = CamadaFullConnect_new(self->gpu, self->queue, self->getSizeOut(self), numero_neuronios,
									 internal_Cnn_getEntrada(self), p, funcaoAtivacao, self->erro, rdp_pesos, rdp_bias);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_Padding(Cnn self, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right) {
	Camada c = CamadaPadding_new(self->gpu, self->queue, self->getSizeOut(self), top, bottom, left, right, internal_Cnn_getEntrada(self), self->erro);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_DropOut(Cnn self, REAL probabilidadeSaida, cl_ulong seed) {
	Camada c = CamadaDropOut_new(self->gpu, self->queue, self->getSizeOut(self), probabilidadeSaida, seed, internal_Cnn_getEntrada(self), self->erro);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_SoftMax(Cnn self) {
	Camada c = CamadaSoftMax_new(self->gpu, self->queue, self->getSizeOut(self), internal_Cnn_getEntrada(self), self->erro);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_BatchNorm(Cnn self, REAL epsilon, Parametros p, RandomParams randY, RandomParams randB) {
	Camada c = CamadaBatchNorm_new(self->gpu, self->queue, p, self->getSizeOut(self), internal_Cnn_getEntrada(self), epsilon, self->erro, randY, randB);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_normalizeIMAGE(Cnn self, Tensor dst, Tensor src) {
	if (self->erro->error)return self->erro->error;
	REAL maximo = -INFINITY;
	REAL minimo = +INFINITY;
	uint8_t *data = src->getvalues(src, NULL);
	for (int i = 0; i < src->length; ++i) {
		if (maximo < data[i]) {
			maximo = data[i];
		}
		if (minimo > data[i]) {
			minimo = data[i];
		}
	}
	free_mem(data);
	maximo = maximo - minimo;
	if (maximo == 0.0)maximo = 1;
	Kernel normalizeChar2Real = ((Kernel *) self->kernels)[CNN_KERNEL_NORMALIZE_CHAR_2_REAL];
	self->erro->error = normalizeChar2Real->runRecursive(normalizeChar2Real, self->queue, src->length, self->gpu->maxworks,
														 &dst->data, &src->data, &maximo, &minimo);
	return self->erro->error;
}

char *Cnn_json(Cnn self, int showValues) {
	char *string = NULL;
	int len = 0;
	P3d sizeout = self->getSizeOut(self);
	apendstr(string, len, "{\n"PAD"\"size_in\":{\"x\":%zu,\"y\":%zu,\"z\":%zu},\n", self->size_in.x, self->size_in.y, self->size_in.z);

	apendstr(string, len, PAD"\"size_out\":{\"x\":%zu,\"y\":%zu,\"z\":%zu}", sizeout.x, sizeout.y, sizeout.z);
	char *tmp = NULL;
	apendTensor("entrada", entrada, string, len, tmp, showValues);
	apendTensor("target", target, string, len, tmp, showValues);
	apendTensor("ds", ds, string, len, tmp, showValues);
	apendstr(string, len, ",\n"PAD "\"numero_camadas\":%zu,\n\"camadas\":[", self->l);
	for (int i = 0; i < self->l; ++i) {
		tmp = self->cm[i]->json(self->cm[i], showValues);
		if (i > 0) {
			apendstr(string, len, ",\n");
		}
		apendstr(string, len, "%s", tmp);
		free_mem(tmp);
	}
	apendstr(string, len, "]\n} ");

	return string;
}

int Cnn_extractVectorLabelClass(Cnn self, Tensor dst, Tensor label) {
	if (self->erro->error)return self->erro->error;
	Kernel extract = ((Kernel *) self->kernels)[CNN_KERNEL_EXTRACT_VECTOR_CLASS_FROM_CHAR];
	self->erro->error = extract->runRecursive(extract, self->queue, label->w, self->gpu->maxworks, &dst->data, &label->data, &label->y);
	return self->erro->error;
}

int Cnn_setInput(Cnn self, size_t x, size_t y, size_t z) {
	if (self->l != 0)return 10;
	P3d size_in = P3D(x, y, z);
	memcpy((void *) &self->size_in, &size_in, sizeof(P3d));
	Release(self->entrada);
	self->entrada = Tensor_new(size_in.x,size_in.y,size_in.z,1,self->erro,0,self->gpu->context,self->queue);
	return 0;
}

void Cnn_jsonF(Cnn self, int showValues, const char *filename) {
	char *json = self->json(self, showValues);
	FILE *f = fopen(filename,"w");
	fprintf(f,"%s\n",json);
	fclose(f);
	free_mem(json);
}

