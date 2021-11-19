//
// Created by hslhe on 18/11/2021.
//

#include "camadas/CamadaBatchNorm.h"


static const char *lname = "BatchNorm";

void CamadaBatchNorm_release(CamadaBatchNorm *selfp) {
	internal_Camada_release((Camada *) (*selfp));
	Release((*selfp)->Y);
	Release((*selfp)->gradY);
	Release((*selfp)->B);
	Release((*selfp)->gradB);
	Release((*selfp)->media);
	Release((*selfp)->somaDiferenca);
	Release((*selfp)->variancia);
	Release((*selfp)->gradVariancia);
	Release((*selfp)->diferenca);
	Release((*selfp)->diferencaquad);
	Release((*selfp)->norma);


	Release((*selfp)->batchNormAtiva1);
	Release((*selfp)->batchNormAtiva2);
	Release((*selfp)->batchNormAtiva3);
	Release((*selfp)->batchNormAtiva4);
	Release((*selfp)->batchNormCalcGrads1);
	Release((*selfp)->batchNormCalcGrads2);

	free_mem(*selfp);
}

int CamadaBatchNorm_propagation(CamadaBatchNorm self) {
	Execute(batchNormAtiva1, self->super.s->z,
			&self->super.a->data, &self->media->data,
			&self->super.a->x, &self->super.a->y
	);
	Execute(batchNormAtiva2,
			self->super.s->z,
			&self->super.a->data, &self->media->data,
			&self->diferenca->data, &self->diferencaquad,
			&self->super.a->x, &self->super.a->y
	);
	Execute(batchNormAtiva3,
			self->super.s->z,
			&self->diferenca->data, &self->diferencaquad->data,
			&self->somaDiferenca->data, &self->variancia->data,
			&self->epsilon, &self->diferenca->x, &self->diferenca->y

	);
	Execute(batchNormAtiva4,
			self->super.s->lenght,
			&self->super.s->data,
			&self->norma->data,
			&self->diferenca->data,
			&self->diferenca->data,
			&self->variancia->data,
			&self->Y->data,
			&self->B->data,
			&self->diferenca->x,
			&self->diferenca->y
	);

	return self->super.erro->error;
}

int CamadaBatchNorm_backpropagation(CamadaBatchNorm self, Tensor ds) {
	if (self->super.da) {
		Execute(batchNormCalcGrads1,
				self->super.s->lenght,
				&self->super.da->data,
				&ds->data,
				&self->variancia->data,
				&self->media->data,
				&self->Y->data,
				&self->somaDiferenca->data,
				&self->super.a->data,
				&self->super.a->x,
				&self->super.a->y
		);
	}
	if (!self->super.params.disable_learn)
		Execute(batchNormCalcGrads2,
				self->super.s->lenght,
				&ds->data,
				&self->norma->data,
				&self->Y->data,
				&self->B->data,
				&self->gradY->data,
				&self->gradB->data,
				&self->super.params.hitlearn,
				&self->super.params.momento,
				&self->super.params.decaimento,
				&self->super.a->x,
				&self->super.a->y
		);
	return self->super.erro->error;
}

char *CamadaBatchNorm_json(CamadaBatchNorm self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{"
					 PAD"%s"
					 PAD"\"epsilon\":%g,\n",
			 tmp,
			 (double) self->epsilon
	)
	free_mem(tmp);
	tmp = self->Y->json(self->Y, showValues);
	apendstr(string, len, PAD"\"Y\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->gradY->json(self->gradY, showValues);
	apendstr(string, len, PAD"\"gradY\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->B->json(self->B, showValues);
	apendstr(string, len, PAD"\"B\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->gradB->json(self->gradB, showValues);
	apendstr(string, len, PAD"\"gradB\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->media->json(self->media, showValues);
	apendstr(string, len, PAD"\"media\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->somaDiferenca->json(self->somaDiferenca, showValues);
	apendstr(string, len, PAD"\"somaDiferenca\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->variancia->json(self->variancia, showValues);
	apendstr(string, len, PAD"\"variancia\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->gradVariancia->json(self->gradVariancia, showValues);
	apendstr(string, len, PAD"\"gradVariancia\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->diferenca->json(self->diferenca, showValues);
	apendstr(string, len, PAD"\"diferenca\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->diferencaquad->json(self->diferencaquad, showValues);
	apendstr(string, len, PAD"\"diferencaquad\":%s,\n", tmp);
	free_mem(tmp);
	tmp = self->norma->json(self->norma, showValues);
	apendstr(string, len, PAD"\"norma\":%s\n}", tmp);
	free_mem(tmp);
	return string;
}

char *CamadaBatchNorm_getGenerate(CamadaBatchNorm self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s(%g,Params(%g,%g,%g,%d),RDP(%d,%g,%g),RDP(%d,%g,%g))",
			 lname,
			 self->epsilon,
			 (double) self->super.params.hitlearn,
			 (double) self->super.params.momento,
			 (double) self->super.params.decaimento,
			 self->super.params.disable_learn,
			 self->rdp_Y.type,
			 (double) self->rdp_Y.a,
			 (double) self->rdp_Y.b,
			 self->rdp_B.type,
			 (double) self->rdp_B.a,
			 (double) self->rdp_B.b
	)
	return string;
}

/**
 * Salva a camada pool em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 byte -> separação deve ser '#'
 * 4 bytes -> size_element REAL
 * 8 bytes -> epsilon
 * 8 bytes -> dimensao z
 * z*size_element -> Y data
 * z*size_element -> B data
 *
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
int CamadaBatchNorm_save(CamadaBatchNorm self, FILE *f) {
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaBatchNorm_save");
	fwrite(&self->super.layer_id, 1, sizeof(int), f);
	fwrite("#", 1, 1, f);
	fwrite(&self->Y->size_element, 1, sizeof(unsigned int), f);
	fwrite(&self->epsilon, 1, sizeof(REAL), f);
	fwrite(&self->Y->z, 1, sizeof(size_t), f);
	void *data = self->Y->getvalues(self->Y, NULL);
	fwrite(data, self->Y->lenght, self->Y->size_element, f);
	data = self->B->getvalues(self->B, data);
	fwrite(data, self->B->lenght, self->B->size_element, f);
	free_mem(data);

	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

extern Camada CamadaBatchNorm_new(Gpu gpu, Queue queue, Parametros params, P3d size_in, Tensor entrada,
								  REAL epsilon, Ecx ecx, RdP randY, RdP randB) {
	ecx->addstack(ecx, "CamadaBatchNorm_new");
	CamadaBatchNorm self = alloc_mem(1, sizeof(CamadaBatchNorm_t));

	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, POOLING_ID, lname, params, entrada, size_in, size_out, ecx);
	self->epsilon = epsilon;
	self->media = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->somaDiferenca = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->variancia = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->gradVariancia = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);

	self->Y = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->B = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->gradY = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->gradB = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);

	self->diferenca = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->diferencaquad = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->norma = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->rdp_Y = randY;
	self->rdp_B = randB;
	if (randY.type != -1) {
		if (randY.type == 0) {
			randY.type = TENSOR_UNIFORM;
			randY.a = (REAL) (2.0) / (REAL) (size_in.x * size_in.y * size_in.z);
			randY.b = -(REAL) 0.5 * randY.a;
		}
		self->super.erro->setError(self->super.erro, self->Y->randomize(self->Y, randY.type, randY.a, randY.b));
	}
	if (randB.type != -1) {
		if (randB.type == 0) {
			randB.type = TENSOR_NORMAL;
			randB.a = (REAL) (2.0) / (REAL) (size_in.x * size_in.y * size_in.z);
			randB.b = 0;
		}
		self->super.erro->setError(self->super.erro, self->B->randomize(self->B, randB.type, randB.a, randB.b));
	}

	//kernels
	self->batchNormAtiva1 = Kernel_news(gpu->program, "BatchNormMedia", "Vector entrada, Vector media,\n"
																		"int entradatx, int entradaty, int k0");
	CheckKernel(batchNormAtiva1);

	self->batchNormAtiva2 = Kernel_news(gpu->program, "BatchNormDiferenca",
										"Vector entrada, Vector media,\n"
										"Vector diferenca,\n"
										"Vector diferencaquad,\n"
										"int entradatx, int entradaty, int k0");
	CheckKernel(batchNormAtiva2);
	self->batchNormAtiva3 = Kernel_news(gpu->program, "BatchNormVariance", "Vector dif, Vector difQuad,\n"
																		   "Vector sumdiferenca, Vector variancia,\n"
																		   "REAL episolon, int diftx, int difty,\n"
																		   "int k0");
	CheckKernel(batchNormAtiva3);
	self->batchNormAtiva4 = Kernel_news(gpu->program, "BatchNormNormaliza",
										"Vector saida,\n"
										"Vector norma,\n"
										"Vector diferenca,\n"
										"Vector variancia,\n"
										"Vector Y,\n"
										"Vector B,\n"
										"int diferencatx, int diferencaty, int k0");
	CheckKernel(batchNormAtiva4);
	self->batchNormCalcGrads1 = Kernel_news(gpu->program, "BatchNormaCalcGrad1", "Vector gradIn,\n"
																				 "Vector gradNext,\n"
																				 "Vector variancia,\n"
																				 "Vector media,\n"
																				 "Vector Y,\n"
																				 "Vector somaDif,\n"
																				 "Vector entrada,\n"
																				 "int entradatx,\n"
																				 "int entradaty,\n"
																				 "int k0");
	CheckKernel(batchNormCalcGrads1);
	self->batchNormCalcGrads2 = Kernel_news(gpu->program, "BatchNormaCalcGrad2",
											"Vector gradNext,\n"
											"Vector norma,\n"
											"Vector Y,\n"
											"Vector B,\n"
											"Vector gradY,\n"
											"Vector gradB,\n"
											"REAL hitlearn,\n"
											"REAL momento,\n"
											"REAL weightDecay,\n"
											"int entradatx,\n"
											"int entradaty,\n"
											"int k0");
	CheckKernel(batchNormCalcGrads2);
	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaBatchNorm_release;
	self->super.propagation = (int (*)(void *)) CamadaBatchNorm_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor *)) CamadaBatchNorm_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaBatchNorm_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaBatchNorm_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaBatchNorm_save;
	return (Camada) self;
}
