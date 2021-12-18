//
// Created by hslhe on 19/11/2021.
//


#include "camadas/CamadaSoftMax.h"

static const char *lname = "SoftMax";

void CamadaSoftMax_release(CamadaSoftMax *selfp) {
	Release((*selfp)->soma);
	Release((*selfp)->exponent);
	Release((*selfp)->softmaxExp);
	Release((*selfp)->softmaxSomaExp);
	Release((*selfp)->softmaxNormaliza);
	Release((*selfp)->softMaxcalcgrad);
	Release((*selfp)->softmaxFindMax);

	Release((*selfp)->maximos);
	Release((*selfp)->indice_maximos);

	internal_Camada_release((Camada *) (selfp));
	gab_free(*selfp);
}

int CamadaSoftMax_propagation(CamadaSoftMax self) {

	if ((self->flag & 0b10) == SOFTNORM) { // é normalizado
		// encontrar o maximo da entrada
		Execute(softmaxFindMax, self->super.s->z,
				&self->super.a->data, &self->maximos->data,
				&self->indice_maximos->data, &self->super.a->x, &self->super.a->y
		);
		//calcular exponencial normalizada
		Execute(softmaxExp, self->super.s->length,
				&self->super.a->data, &self->exponent->data,
				&self->maximos->data, &self->super.a->x, &self->super.a->y
		);
	} else {// não é normalizado
		// faz a exponencial da entrada
		Execute(softmaxExp, self->super.s->length,
				&self->super.a->data, &self->exponent->data
		);
	}
	// soma as exponenciais
	Execute(softmaxSomaExp, self->super.s->z,
			&self->exponent->data, &self->soma->data,
			&self->super.s->x, &self->super.s->y
	);
	// divide as exponenciais e armazena na saída
	Execute(softmaxNormaliza, self->super.s->length,
			&self->exponent->data, &self->soma->data,
			&self->super.s->data,
			&self->super.s->x, &self->super.s->y
	);
	return self->super.ecx->error;
}

int CamadaSoftMax_backpropagation(CamadaSoftMax self, Tensor ds) {
	if (self->super.da) {// tem que calcular o gradiente de entrada
		if ((self->flag & 0b1) == SOFTLAST) {// é a ultima camada
			// faz a copia de ds para da. Onde da =  s - t
			self->super.da->copy(self->super.da, ds);
		}
	} else {// não é a ultima camada
		// ds vai ser escrito
		// se não for normalizado ds e da são os mesmos
		// calcula a derivada da camada usando o jacobiano
		Execute(softMaxcalcgrad, self->super.da->length,
				&ds, &self->super.s->data,
				&ds->data, &self->super.s->x, &self->super.s->y
		);
	}

	return self->super.ecx->error;
}

char *CamadaSoftMax_json(CamadaSoftMax self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{"
					 PAD"%s",
			 tmp);
	gab_free(tmp);
	apendTensor("exponencial", exponent, string, len, tmp, showValues);
	apendTensor("soma", soma, string, len, tmp, showValues);
	apendstr(string, len, "\n}");

	return string;
}

char *CamadaSoftMax_getGenerate(CamadaSoftMax self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s (%d)",
			 lname, (int) self->flag
	);
	return string;
}

/**
 * Salva a camada softMax em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 byte -> separação deve ser '#'
 *
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
int CamadaSoftMax_save(CamadaSoftMax self, FILE *f) {
	if (self->super.ecx->error)goto end;
	ECXPUSH(self->super.ecx);
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->flag, 1, 1, f);
	end:
	ECXPOP(self->super.ecx);
	return self->super.ecx->error;
}

Camada CamadaSoftMax_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ECXPUSH(ecx);
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;
	internal_loadCamada(f, &parametros, &size_in, &size_element);;
	char flag = 0;
	fread(&flag, 1, 1, f);
	CamadaSoftMax self = (CamadaSoftMax) CamadaSoftMax_new(gpu, queue, flag, size_in, entrada, ecx);
	end:
	ECXPOP(ecx);
	return (Camada) self;
}

Camada CamadaSoftMax_new(Gpu gpu, Queue queue, char flag, P3d size_in, Tensor entrada, Ecx ecx) {
	ECXPUSH(ecx);
	CamadaSoftMax self = gab_alloc(1, sizeof(CamadaSoftMax_t));

	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, SOFTMAX_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);

	self->soma = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->exponent = Tensor_new(size_in.x, size_in.y, size_in.z, 1, ecx, 0, gpu->context, queue);
	memcpy((void *) &self->flag, &flag, sizeof(char));
	if ((flag & 0b10) == SOFTNORM) {// é normalizado, implica em ter os tensores maximo e index maximo
		self->maximos = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
		self->indice_maximos = Tensor_new(1, 1, size_in.z, 1, ecx, TENSOR_INT, gpu->context, queue);

		self->softmaxExp = Kernel_news(gpu->program, "softmaxExpNorm", "Vector entrada, Vector exponent,Vector mx,int ax,int ay, int k0");
		self->softmaxFindMax = Kernel_news(gpu->program, "softmaxFindMax", "Vector a, Vector mx, __global int *i_max, int ax, int ay, int k0");
	} else {
		self->softmaxExp = Kernel_news(gpu->program, "softmaxExp", "Vector entrada, Vector exponent,int k0");
	}
	CheckKernel(softmaxExp);
	self->softmaxSomaExp = Kernel_news(gpu->program, "softmaxSomaExp", "Vector eps, Vector soma, int saidatx, int saidaty, int k0");
	CheckKernel(softmaxSomaExp);
	self->softmaxNormaliza = Kernel_news(gpu->program, "softmaxNormaliza", "Vector exponet, Vector soma, Vector saida,int saidatx, int saidaty, int k0");
	CheckKernel(softmaxNormaliza);
	if ((flag & 0b1) != SOFTLAST) {// não é a ultima camada, tem que calcular o jacobiano
		self->softMaxcalcgrad = Kernel_news(gpu->program, "softMaxcalcgrad", "Vector da, Vector s, Vector ds, int sx, int sy, int k0");
		CheckKernel(softMaxcalcgrad);
	}
	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaSoftMax_release;
	self->super.propagation = (int (*)(void *)) CamadaSoftMax_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaSoftMax_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaSoftMax_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaSoftMax_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaSoftMax_save;
	return (Camada) self;
}
