//
// Created by hslhe on 18/11/2021.
//

#include "camadas/CamadaBatchNorm.h"


static const char *lname = "BatchNorm";

void CamadaBatchNorm_release(CamadaBatchNorm *selfp) {
	internal_Camada_release((Camada *) (selfp));
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
			self->super.s->length,
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

	return self->super.ecx->error;
}

int CamadaBatchNorm_backpropagation(CamadaBatchNorm self, Tensor ds) {
	if (self->super.da) {
		Execute(batchNormCalcGrads1,
				self->super.s->length,
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
	if (!self->super.params.skipLearn)
		Execute(batchNormCalcGrads2,
				self->super.s->length,
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
	return self->super.ecx->error;
}

char *CamadaBatchNorm_json(CamadaBatchNorm self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{"
					 PAD"%s,\n"
					 PAD"\"epsilon\":%g",
			 tmp,
			 (double) self->epsilon
	);
	free_mem(tmp);
	apendTensor("Y",Y,string,len,tmp,showValues);
	apendTensor("dY",gradY,string,len,tmp,showValues);
	apendTensor("B",B,string,len,tmp,showValues);
	apendTensor("dB",gradB,string,len,tmp,showValues);
	apendTensor("media",media,string,len,tmp,showValues);
	apendTensor("somaDiferenca",somaDiferenca,string,len,tmp,showValues);
	apendTensor("variancia",variancia,string,len,tmp,showValues);
	apendTensor("gradVariancia",gradVariancia,string,len,tmp,showValues);
	apendTensor("diferenca",diferenca,string,len,tmp,showValues);
	apendTensor("diferencaquad",diferencaquad,string,len,tmp,showValues);
	apendTensor("norma",norma,string,len,tmp,showValues);
	apendstr(string, len,"\n}");
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
			 self->super.params.skipLearn,
			 self->rdp_Y.type,
			 (double) self->rdp_Y.a,
			 (double) self->rdp_Y.b,
			 self->rdp_B.type,
			 (double) self->rdp_B.a,
			 (double) self->rdp_B.b
	);
	return string;
}

/**
 * Salva a camada batchnorm em um arquivo
 * Camada
 * epsilon
 * Y
 * B
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
int CamadaBatchNorm_save(CamadaBatchNorm self, FILE *f) {
	if (self->super.ecx->error)goto end;
	self->super.ecx->addstack(self->super.ecx, "CamadaBatchNorm_save");
	internal_saveCamada(f, (Camada) self);
	internal_saveREAL(f,self->epsilon);
	internal_saveTensor(f,self->Y);
	internal_saveTensor(f,self->B);
	end:
	self->super.ecx->popstack(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaBatchNorm_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	if (ecx->error)goto end;
	ecx->addstack(ecx, "CamadaBatchNorm_load");
	P3d size_in;
	Parametros  parametros;
	uint32_t size_element;
	REAL epsilon;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	internal_loadREAL(f,&epsilon,size_element);
	CamadaBatchNorm  self = (CamadaBatchNorm) CamadaBatchNorm_new(gpu, queue, parametros, size_in, entrada, epsilon, ecx, RDP(-1), RDP(-1));
	internal_loadTensor(f,self->Y,size_element);
	internal_loadTensor(f,self->B,size_element);
	end:
	ecx->popstack(ecx);
	return (Camada) self;
}

extern Camada CamadaBatchNorm_new(Gpu gpu, Queue queue, Parametros params, P3d size_in, Tensor entrada,
								  REAL epsilon, Ecx ecx, RandomParams randY, RandomParams randB) {
	ecx->addstack(ecx, "CamadaBatchNorm_new");
	CamadaBatchNorm self = alloc_mem(1, sizeof(CamadaBatchNorm_t));

	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, BATCHNORM_ID, lname, params, entrada, size_in, size_out, ecx);
	self->epsilon = epsilon;
	self->media = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->somaDiferenca = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->variancia = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->gradVariancia = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);

	self->Y = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->B = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->gradY = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->gradB = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->gradY->fill(self->gradY,0);
	self->gradB->fill(self->gradB,0);
	self->diferenca = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->diferencaquad = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->norma = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->rdp_Y = randY;
	self->rdp_B = randB;
	if (randY.type != -1) {

		if (randY.type == 0) {
			randY = internal_getDefaultRDP(0, self->super.a->length, self->super.s->length);
//			randY.type = TENSOR_UNIFORM;
//			randY.a = (REAL) (2.0) / (REAL) (size_in.x * size_in.y * size_in.z);
//			randY.b = -(REAL) 0.5 * randY.a;
		}
		self->super.ecx->setError(self->super.ecx, self->Y->randomize(self->Y, randY.type, randY.a, randY.b));
	}
	if (randB.type != -1) {
		if (randB.type == 0) {
			randB = internal_getDefaultRDP(1, self->super.a->length, self->super.s->length);
//			randB.type = TENSOR_GAUSSIAN;
//			randB.a = (REAL) (2.0) / (REAL) (size_in.x * size_in.y * size_in.z);
//			randB.b = 0;
		}
		self->super.ecx->setError(self->super.ecx, self->B->randomize(self->B, randB.type, randB.a, randB.b));
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
	self->super.retroPropagation = (int (*)(void *, Tensor )) CamadaBatchNorm_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaBatchNorm_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaBatchNorm_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaBatchNorm_save;
	return (Camada) self;
}
