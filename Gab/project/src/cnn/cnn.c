//
// Created by Henrique on 21/11/2021.
//

#include <math.h>
#include <error_list.h>
#include "cnn/cnn.h"
#include "cnn/kernel_lib.h"
#include "camadas/all_camadas.h"

#define NKERNELS 3
#define CNN_KERNEL_SUB 0
#define CNN_KERNEL_NORMALIZE_CHAR_2_REAL 1
#define CNN_KERNEL_EXTRACT_VECTOR_CLASS_FROM_CHAR 2


#define CHECKDIN(input, filtro, abertura, passo) \
    (((((input-1) - (filtro - 1) * abertura) / passo +1)>0) && \
    (((((input-1) - (filtro - 1) * abertura) / passo)*passo + (filtro-1)*abertura) == (input-1)))
const char *versao = "3.0.0012";

const char *Cnn_version() {
	return versao;
}


//#####################################################################################
//							FUNÇÕES PARA ADICIONA CAMADAS
//#####################################################################################


int Cnn_release(Cnn *selfp) {
	if (!selfp) {
		return 10;
	}
	if (!(*selfp)) {
		return 10;
	}
	(*selfp)->ecx->print((*selfp)->ecx);
	if ((*selfp)->ecx->error) {
		char *tmp = Gpu_errormsg((*selfp)->ecx->error);
		fprintf(stderr, "%s\n", tmp);
		gab_free(tmp);
	}
	int erro = (*selfp)->ecx->error;
	for (int l = 0; l < (*selfp)->l; ++l) {
		(*selfp)->cm[l]->release(&(*selfp)->cm[l]);
	}
	if ((*selfp)->cm) {
		gab_free((*selfp)->cm);
	}

	Release((*selfp)->entrada);
	Release((*selfp)->target);
	Release((*selfp)->ds);
	Kernel *kernels = (*selfp)->kernels;
	if ((*selfp)->kernels) {
		for (int i = 0; i < NKERNELS; ++i) {
			Release(kernels[i]);
		}

		gab_free((*selfp)->kernels);
	}
	if ((*selfp)->release_gpu) {
		Release((*selfp)->gpu);
	}
	if ((*selfp)->releaseL) {
		(*selfp)->releaseL((*selfp)->LuaVm);
	}
	gab_free(*selfp);
	return erro;
}

void internal_Cnn_getKernels(Cnn self) {
	Kernel sub = Kernel_news(self->gpu->program, "kernel_sub", "Vector ds, Vector s, Vector t, int k0");
	Kernel normalizechar2real = Kernel_news(self->gpu->program, "kernel_normalizechar2real", "Vector dst, __global char *src, REAL a, REAL b, int k0");
	self->ecx->setError(self->ecx, normalizechar2real->error, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
	Kernel getVetorClassFromChar = Kernel_news(self->gpu->program, "kernel_getVetorClassFromChar", " Vector dst, __global unsigned char *ints,unsigned int noptiobs, int k0");
	self->ecx->setError(self->ecx, normalizechar2real->error, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
	Kernel *allkernels = gab_alloc(NKERNELS, sizeof(Kernel));
	self->ecx->setError(self->ecx, normalizechar2real->error, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);

	allkernels[CNN_KERNEL_SUB] = sub;
	allkernels[CNN_KERNEL_NORMALIZE_CHAR_2_REAL] = normalizechar2real;
	allkernels[CNN_KERNEL_EXTRACT_VECTOR_CLASS_FROM_CHAR] = getVetorClassFromChar;
	self->kernels = allkernels;
}

int internal_Cnn_addlayer(Cnn self, Camada newLayer) {
	if (self->ecx->error) {
		newLayer->release(newLayer);
		return self->ecx->error;
	}
	if (self->lock) {
		fprintf(stderr, "Uma camada com ativação softmax foi utilizada, não é possivel adicionar mais camdas.\n");
		self->ecx->pushMsg(self->ecx, "Uma camada com ativação softmax foi utilizada, não é possivel adicionar mais camdas.");
		newLayer->release(newLayer);
		self->ecx->error = GAB_INVALID_LAYER;
		return self->ecx->error;
	}
	self->l = self->l + 1;
	self->cm = realloc(self->cm, self->l * sizeof(Camada));
	self->cm[self->l - 1] = newLayer;
	Release(self->target);
	Release(self->ds);
	P3d size_out = newLayer->getOutSize(newLayer);
	self->target = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->ecx, 0, self->gpu->context, self->queue);
	self->ds = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->ecx, 0, self->gpu->context, self->queue);
	return self->ecx->error;
}

Tensor internal_Cnn_getEntrada(Cnn self) {
	if (self->cm) {
		return self->cm[self->l - 1]->s;
	}
	return NULL;
}

P3d Cnn_getSizeOut(Cnn self) {
	if (self->l > 0) {
		return self->cm[self->l - 1]->getOutSize(self->cm[self->l - 1]);
	}
	return self->size_in;
}

void Cnn_removeLastLayer(Cnn self) {
	if (self->l <= 0) {
		return;
	}
	self->lock = 0;
	self->l = self->l - 1;
	Release(self->cm[self->l]);
	if (self->l == 0) {
		gab_free(self->cm);
		self->cm = NULL;
	} else {
		self->cm = gab_realloc(self->cm, self->l * sizeof(Camada));
	}

	Release(self->target);
	Release(self->ds);
	P3d size_out = self->getSizeOut(self);
	self->target = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->ecx, 0, self->gpu->context, self->queue);
	self->ds = Tensor_new(size_out.x, size_out.y, size_out.z, 1, self->ecx, 0, self->gpu->context, self->queue);
}

int Cnn_learn(Cnn self, Tensor target);

int Cnn_predict(Cnn self, Tensor entrada);

int Cnn_learnv(Cnn self, REAL *target) {
	if (!target) {
		self->ecx->error = GAB_NULL_POINTER_ERROR;
	}
	if (!self->l) {
		self->ecx->error = GAB_CNN_NOT_INITIALIZED;
	}
	if (self->ecx->error) {
		return self->ecx->error;
	}
	self->target->setvalues(self->target, target);
	return Cnn_learn(self, self->target);
}

int Cnn_predictv(Cnn self, REAL *entrada) {
	if (!entrada) {
		self->ecx->error = GAB_NULL_POINTER_ERROR;
	}
	if (!self->l) {
		self->ecx->error = GAB_CNN_NOT_INITIALIZED;
	}
	if (self->ecx->error) {
		return self->ecx->error;
	}
	self->entrada->setvalues(self->entrada, entrada);
	return Cnn_predict(self, self->entrada);

}

int Cnn_learn(Cnn self, Tensor target) {
	if (!self->l) {
		self->ecx->error = GAB_CNN_NOT_INITIALIZED;
	}
	if (self->ecx->error) {
		return self->ecx->error;
	}
	if (target->flag.ram || target->flag.shared) {
		return Cnn_learnv(self, target->data);
	}
	Kernel sub = ((Kernel *) self->kernels)[CNN_KERNEL_SUB];
	sub->runRecursive(sub, self->queue, self->ds->length, self->gpu->maxworks, &self->ds->data, &self->cm[self->l - 1]->s->data, &target->data);
	Tensor ds = self->ds;
	for (int l = self->l - 1; l >= 0 && !self->ecx->error; l--) {
		self->cm[l]->retroPropagation(self->cm[l], ds);
		ds = self->cm[l]->da;
	}
	return self->ecx->error;
}

int Cnn_learnBatch(Cnn self, Tensor target, size_t batchSize) {
	if (!self->l) {
		self->ecx->error = GAB_CNN_NOT_INITIALIZED;
	}
	if (self->ecx->error) {
		return self->ecx->error;
	}
	if (target->flag.ram || target->flag.shared) {
		self->target->setvalues(self->target, target->data);
		return Cnn_learnBatch(self, self->target, batchSize);
	}
	Kernel sub = ((Kernel *) self->kernels)[CNN_KERNEL_SUB];
	sub->runRecursive(sub, self->queue, self->ds->length, self->gpu->maxworks, &self->ds->data, &self->cm[self->l - 1]->s->data, &target->data);
	Tensor ds = self->ds;


	for (int l = self->l - 1; l >= 0 && !self->ecx->error; l--) {
		self->cm[l]->retroPropagationBatch(self->cm[l], ds, batchSize);
		ds = self->cm[l]->da;
	}
	return self->ecx->error;
}

int Cnn_fixBatch(Cnn self) {
	if (!self->l) {
		self->ecx->error = GAB_CNN_NOT_INITIALIZED;
	}
	if (self->ecx->error) {
		return self->ecx->error;
	}
	for (int l = self->l - 1; l >= 0 && !self->ecx->error; l--) {
		self->cm[l]->retroPropagationBatchLearn(self->cm[l]);
	}
	return self->ecx->error;
}


int Cnn_predict(Cnn self, Tensor entrada) {
	if (!self->l) {
		self->ecx->error = GAB_CNN_NOT_INITIALIZED;
	}
	if (self->ecx->error) {
		return self->ecx->error;
	}
	if (entrada->flag.ram || entrada->flag.shared) {
		return Cnn_predictv(self, entrada->data);
	}
	Tensor a = entrada;
	for (int l = 0; l < self->l && !self->ecx->error; ++l) {
		a = self->cm[l]->propagation(self->cm[l], a);
	}
	return self->ecx->error;
}

int Cnn_updateHitLearn(Cnn self, size_t iter) {
	for (int i = 0; i < self->l; ++i) {
		self->cm[i]->updateHitLearn(self->cm[i], iter);
	}
	return 0;
}

REAL Cnn_mse(Cnn self) {
	if (self->ecx->error) {
		return NAN;
	}
	if (self->l <= 0) {
		return NAN;
	}
	double mse = 0;
	double tmp;
	REAL *data = self->ds->getvalues(self->ds, NULL);

	for (int i = 0; i < self->ds->length; ++i) {
		tmp = data[i];
		mse += (double) tmp * tmp;
	}
	gab_free(data);
	return mse / 2;

}

REAL Cnn_mseT(Cnn self, Tensor target) {
	if (self->l <= 0) {
		return NAN;
	}
	REAL *data = self->cm[self->l - 1]->s->getvalues(self->ds, NULL);
	REAL *dataT = target->getvalues(target, NULL);
	REAL mse = 0;
	REAL tmp;
	for (int i = 0; i < self->cm[self->l - 1]->s->length; ++i) {
		tmp = data[i] - dataT[i];
		mse += tmp * tmp;
	}
	gab_free(data);
	gab_free(dataT);
	return mse / 2;
}

int Cnn_maxIndex(Cnn self) {
	if (self->l <= 0) {
		return 0;
	}
	if(self->ecx->error){
		fprintf(stderr,"Erro %d %s:%d\n",self->ecx->error,__FILE__,__LINE__);
		return 0;
	}
	REAL *data = self->cm[self->l - 1]->s->getvalues(self->cm[self->l - 1]->s, NULL);
	int mindex = 0;
	for (int i = 1; i < self->cm[self->l - 1]->s->length; ++i) {
		if (data[i] > data[mindex]) {
			mindex = i;
		}
	}
	gab_free(data);
	return mindex;
}

int Cnn_normalizeIMAGE(Cnn self, Tensor dst, Tensor src) {
	if (self->ecx->error) {
		return self->ecx->error;
	}
	REAL maximo = 255.0;
	REAL minimo = 0;

	Kernel normalizeChar2Real = ((Kernel *) self->kernels)[CNN_KERNEL_NORMALIZE_CHAR_2_REAL];
	self->ecx->error = normalizeChar2Real->runRecursive(normalizeChar2Real, self->queue, src->length, self->gpu->maxworks, &dst->data, &src->data, &maximo, &minimo);
	clFinish(self->queue);
	return self->ecx->error;
}

int Cnn_setInput(Cnn self, size_t x, size_t y, size_t z) {
	if (self->l != 0) {
		return 10;
	}
	P3d size_in = P3D(x, y, z);
	memcpy((void *) &self->size_in, &size_in, sizeof(P3d));
	Release(self->entrada);
	self->entrada = Tensor_new(size_in.x, size_in.y, size_in.z, 1, self->ecx, 0, self->gpu->context, self->queue);
	return 0;
}


//#####################################################################################
//
//						FUNÇÕES DE SALVAR E CARREGAR
//
//#####################################################################################

int Cnn_save(Cnn self, const char *filename) {
	if (self->ecx->error) {
		return self->ecx->error;
	}
	FILE *f = fopen(filename, "wb");
	fwrite(&self->size_in, sizeof(P3d), 1, f);
	fwrite(&self->l, sizeof(size_t), 1, f);
	for (int i = 0; i < self->l && !self->ecx->error; ++i) {
		self->cm[i]->save(self->cm[i], f);
	}
	fclose(f);
	return self->ecx->error;
}

int Cnn_load(Cnn self, const char *filename) {
	FILE *f = fopen(filename, "rb");
	Camada c;
	P3d sizeoutcnn;
	P3d sizeinlayer;
	char layerid;
	size_t l;
	fread(&sizeoutcnn, sizeof(P3d), 1, f);
	fread(&l, sizeof(size_t), 1, f);
	self->setInput(self, sizeoutcnn.x, sizeoutcnn.y, sizeoutcnn.z);
	for (int i = 0; i < l; ++i) {
		fread(&layerid, sizeof(char), 1, f);
		sizeoutcnn = self->getSizeOut(self);
		switch (layerid) {
			case CONVOLUCAOF_ID:
				c = CamadaConvF_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case POOL_ID:
				c = CamadaPool_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case FULLCONNECT_ID:
				c = CamadaFullConnect_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case PADDING_ID:
				c = CamadaPadding_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case DROPOUT_ID:
				c = CamadaDropOut_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case RELU_ID:
				c = CamadaRelu_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case PRELU_ID:
				c = CamadaPRelu_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case SOFTMAX_ID:
				c = CamadaSoftMax_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			case BATCHNORM_ID:
				c = CamadaBatchNorm_load(f, self->gpu, self->queue, internal_Cnn_getEntrada(self), self->ecx);
				break;
			default:
				c = NULL;
		}

		if (c == NULL) {
			self->ecx->setError(self->ecx, 46, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
			break;
		}
		sizeinlayer = c->size_in;
		if (sizeinlayer.x != sizeoutcnn.x || sizeinlayer.y != sizeoutcnn.y || sizeinlayer.z != sizeoutcnn.z) {
			self->ecx->setError(self->ecx, 47, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
			break;
		}
		internal_Cnn_addlayer(self, c);
	}

	fclose(f);
	return self->ecx->error;
}

//#####################################################################################
//
//						FUNÇÕES DE print
//
//#####################################################################################

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
		gab_free(tmp);
	}
	apendstr(string, len, "]\n} ");

	return string;
}

int Cnn_extractVectorLabelClass(Cnn self, Tensor dst, Tensor label) {
	if (self->ecx->error) {
		return self->ecx->error;
	}
	dst->fill(dst, 0);
	Kernel extract = ((Kernel *) self->kernels)[CNN_KERNEL_EXTRACT_VECTOR_CLASS_FROM_CHAR];
	self->ecx->error = extract->runRecursive(extract, self->queue, dst->w, self->gpu->maxworks, &dst->data, &label->data, &dst->y);
	return self->ecx->error;
}

void Cnn_jsonF(Cnn self, int showValues, const char *filename) {
	char *json = self->json(self, showValues);
	FILE *f = fopen(filename, "w");
	fprintf(f, "%s\n", json);
	fclose(f);
	gab_free(json);
}

void Cnn_fprint(Cnn self, FILE *f, const char *comment) {
	char *tmp;
	if (comment == NULL) {
		comment = "//";
	}
	P3d out = self->size_in;
	fprintf(f, "Entrada(%zu,%zu,%zu)\n", out.x, out.y, out.z);
	for (int i = 0; i < self->l; ++i) {
		tmp = self->cm[i]->getGenerate(self->cm[i]);
		out = self->cm[i]->getOutSize(self->cm[i]);
		fprintf(f, "%s %s saida [%zu,%zu,%zu]", tmp, comment, out.x, out.y, out.z);
//		switch (self->cm[i]->layer_id) {
//			case CONVOLUCAOF_ID:
//				fprintf(f, "std = %f, u = %f", ((CamadaConvF) self->cm[i])->w->std(((CamadaConvF) self->cm[i])->w), ((CamadaConvF) self->cm[i])->w->media(((CamadaConvF) self->cm[i])->w));
//				break;
//			case FULLCONNECT_ID:
//				fprintf(f, "std = %f, u = %f", ((CamadaFullConnect) self->cm[i])->w->std(((CamadaFullConnect) self->cm[i])->w), ((CamadaFullConnect) self->cm[i])->w->media(((CamadaFullConnect) self->cm[i])->w));
//				break;
//		}
		fprintf(f, "\n");
		gab_free(tmp);
	}
}

void Cnn_mode(Cnn self, int isTraining) {
	isTraining = isTraining != 0;
	for (int l = 0; l < self->l; ++l) {
		if (self->cm[l]->layer_id == DROPOUT_ID) {
			((CamadaDropOut) self->cm[l])->setMode((CamadaDropOut) self->cm[l], isTraining);
		}
	}
	self->mode = isTraining;

}


void Cnn_print(Cnn self, const char *comment) {
	Cnn_fprint(self, stdout, comment);
}

char *Cnn_printstr(Cnn self, const char *comment) {
	char *tmp;

	P3d out = self->size_in;
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "Entrada(%zu,%zu,%zu)\n", out.x, out.y, out.z);
	for (int i = 0; i < self->l; ++i) {
		tmp = self->cm[i]->getGenerate(self->cm[i]);
		out = self->cm[i]->getOutSize(self->cm[i]);
		if (comment) {
			apendstr(string, len, "%s %s saida [%zu,%zu,%zu]\n", tmp, comment, out.x, out.y, out.z);
		} else {
			apendstr(string, len, "%s\n", tmp);
		}
		gab_free(tmp);
	}
	return string;
}


//#####################################################################################
//
//						FUNÇÕES DE ADICIONAR CAMADAS
//
//#####################################################################################


int Cnn_ConvolucaoF(Cnn self, P2d passo, P3d filtro, FAtivacao_t funcaoAtivacao, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right, Parametros p, RandomParams filtros) {
	FAtivacao  fa = {.mask = funcaoAtivacao};
	if(!(fa.id == FSIGMOID|| fa.id == FTANH|| fa.id == FLRELU|| fa.id == FLIN|| fa.id == FALAN|| fa.id == FRELU)){
		self->ecx->setError(self->ecx,GAB_INVALID_PARAM,"Função de ativação desconhecida\n");
		return GAB_INVALID_PARAM;
	}
	P3d size_in = self->getSizeOut(self);
	if (!CHECKDIN(size_in.x+top+bottom, filtro.x, 1, passo.x) || !CHECKDIN(size_in.y+left+right, filtro.y, 1, passo.y)) {
		fprintf(stderr, "ConvolucaoF:Invalid params\nsize in : %zu %zu %zu\nsize out : %g %g %zu\n", size_in.x, size_in.y, size_in.z, (size_in.x - 1 - (filtro.x - 1)) / (REAL) passo.x + 1, (size_in.y - 1 - (filtro.y - 1)) / (REAL) passo.y + 1, size_in.z);
		return GAB_INVALID_PARAM;
	}
	Camada c = CamadaConvF_new(self->gpu, self->queue, size_in, internal_Cnn_getEntrada(self), self->ecx, passo, filtro, funcaoAtivacao, top, bottom, left, right, p, filtros);

	return internal_Cnn_addlayer(self, c);
}




int Cnn_Pooling(Cnn self, P2d passo, P2d filtro, uint32_t type) {
	P3d size_in = self->getSizeOut(self);
	if (!CHECKDIN(size_in.x, filtro.x, 1, passo.x) || !CHECKDIN(size_in.y, filtro.y, 1, passo.y)) {
		fprintf(stderr, "Pooling:Invalid params\nsize in : %zu %zu %zu\nsize out : %g %g %zu\n", size_in.x, size_in.y, size_in.z, (size_in.x - 1 - (filtro.x - 1)) / (REAL) passo.x + 1, (size_in.y - 1 - (filtro.y - 1)) / (REAL) passo.y + 1, size_in.z);
		return GAB_INVALID_PARAM;
	}
	Camada c = CamadaPool_new(self->gpu, self->queue, passo, filtro, size_in, type, internal_Cnn_getEntrada(self), self->ecx);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_Relu(Cnn self, REAL fator_menor0, REAL fator_maior0) {
	Camada c = CamadaRelu_new(self->gpu, self->queue, self->getSizeOut(self), fator_menor0, fator_maior0, internal_Cnn_getEntrada(self), self->ecx);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_PRelu(Cnn self, Parametros params, RandomParams rdp_a) {
	Camada c = CamadaPRelu_new(self->gpu, self->queue, self->getSizeOut(self), internal_Cnn_getEntrada(self), params, rdp_a, self->ecx);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_FullConnect(Cnn self, size_t numero_neuronios, Parametros p, FAtivacao_t funcaoAtivacao, RandomParams rdp_pesos, RandomParams rdp_bias) {
	FAtivacao  fa = {.mask = funcaoAtivacao};
	if(!(fa.id == FSIGMOID|| fa.id == FTANH|| fa.id == FLRELU|| fa.id == FLIN|| fa.id == FALAN|| fa.id == FRELU|| fa.id == FSOFTMAX)){
		self->ecx->setError(self->ecx,GAB_INVALID_PARAM,"Função de ativação desconhecida\n");
		return GAB_INVALID_PARAM;
	}
	Camada c = CamadaFullConnect_new(self->gpu, self->queue, self->getSizeOut(self), numero_neuronios, internal_Cnn_getEntrada(self), p, funcaoAtivacao, self->ecx, rdp_pesos, rdp_bias);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_Padding(Cnn self, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right) {
	Camada c = CamadaPadding_new(self->gpu, self->queue, self->getSizeOut(self), top, bottom, left, right, internal_Cnn_getEntrada(self), self->ecx);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_DropOut(Cnn self, REAL probabilidadeJogarFora, cl_ulong seed) {
	probabilidadeJogarFora = probabilidadeJogarFora>0?probabilidadeJogarFora:-probabilidadeJogarFora;
	if (probabilidadeJogarFora>1){
		return GAB_INVALID_PARAM;
	}
	#if DROPOUT_AS_FIRST_LAYER == 0
	if (self->l <= 0) {
		return GAB_INVALID_LAYER;
	}
	#endif
	Camada c = CamadaDropOut_new(self->gpu, self->queue, self->getSizeOut(self), 1-probabilidadeJogarFora, seed, internal_Cnn_getEntrada(self), self->ecx);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_SoftMax(Cnn self, int8_t flag) {
	Camada c = CamadaSoftMax_new(self->gpu, self->queue, flag, self->getSizeOut(self), internal_Cnn_getEntrada(self), self->ecx);
	return internal_Cnn_addlayer(self, c);
}

int Cnn_BatchNorm(Cnn self, size_t batch_size, REAL epsilon, Parametros p, RandomParams randY, RandomParams randB) {
	Camada c = CamadaBatchNorm_new(self->gpu, self->queue, self->getSizeOut(self), internal_Cnn_getEntrada(self), self->ecx, p, epsilon, batch_size, randY, randB);
	return internal_Cnn_addlayer(self, c);
}


Cnn Cnn_new() {
	Cnn self = gab_alloc(1, sizeof(Cnn_t));
	self->ecx = Ecx_new(0);
	memcpy(&self->version, &versao, sizeof(char *));

	self->gpu = Gpu_new();
	self->ecx->error = self->gpu->compileProgram(self->gpu, (char *) KERNEL_LIB_get_defaultKernel());
	self->release_gpu = 1;


	self->queue = self->gpu->Queue_new(self->gpu, self->ecx->perro);
	internal_Cnn_getKernels(self);
	methods:
	self->ConvolucaoF = Cnn_ConvolucaoF;
	self->Pooling = Cnn_Pooling;
	self->Relu = Cnn_Relu;
	self->PRelu = Cnn_PRelu;
	self->FullConnect = Cnn_FullConnect;
	self->Padding = Cnn_Padding;
	self->DropOut = Cnn_DropOut;
	self->SoftMax = Cnn_SoftMax;
	self->BatchNorm = Cnn_BatchNorm;

	self->print = Cnn_print;
	self->fprint = Cnn_fprint;
	self->printstr = Cnn_printstr;
	self->load = Cnn_load;
	self->save = Cnn_save;
	self->predict = Cnn_predict;
	self->predictv = Cnn_predictv;
	self->learn = Cnn_learn;
	self->learnv = Cnn_learnv;
	self->learnBatch = Cnn_learnBatch;
	self->fixBatch = Cnn_fixBatch;

	self->updateHitLearn = Cnn_updateHitLearn;
	self->mse = Cnn_mse;
	self->mseT = Cnn_mseT;
	self->maxIndex = Cnn_maxIndex;
	self->normalizeIMAGE = Cnn_normalizeIMAGE;
	self->extractVectorLabelClass = Cnn_extractVectorLabelClass;
	self->json = Cnn_json;
	self->jsonF = Cnn_jsonF;
	self->setInput = Cnn_setInput;

	self->removeLastLayer = Cnn_removeLastLayer;
	self->getSizeOut = Cnn_getSizeOut;
	self->setMode = Cnn_mode;
	self->release = Cnn_release;

	return self;
}