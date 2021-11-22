//
// Created by hslhe on 19/11/2021.
//


#include "camadas/CamadaSoftMax.h"

static const char *lname = "SoftMax";

void CamadaSoftMaxing_release(CamadaSoftMax *selfp) {
	internal_Camada_release((Camada *) (*selfp));


	Release((*selfp)->soma);
	Release((*selfp)->exponent);
	Release((*selfp)->softmaxExp);
	Release((*selfp)->softmaxSomaExp);
	Release((*selfp)->softmaxNormaliza);
	Release((*selfp)->softMaxcalcgrad);


	free_mem(*selfp);
}

int CamadaSoftMaxing_propagation(CamadaSoftMax self) {
	Execute(softmaxExp, self->super.s->length,
			&self->super.a->data, &self->exponent->data
	);
	Execute(softmaxSomaExp, self->super.s->length,
			&self->exponent->data, &self->soma->data,
			&self->super.s->x, &self->super.s->y
	);
	Execute(softmaxNormaliza, self->super.s->length,
			&self->exponent->data, &self->soma->data,
			&self->super.s->data,
			&self->super.s->x, &self->super.s->y
	);
	return self->super.erro->error;
}

int CamadaSoftMaxing_backpropagation(CamadaSoftMax self, Tensor ds) {
	if (self->super.da) {
		Execute(softMaxcalcgrad, self->super.da->length,
				&self->super.da->data, &self->super.a->data,
				&ds->data

		);
	}
	return self->super.erro->error;
}

char *CamadaSoftMaxing_json(CamadaSoftMax self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp ;
	apendstr(string, len,
			 "{"
			 PAD"\"size\":%zu"
			 , sizeof(CamadaSoftMax_t));
	apendTensor("exponencial",exponent,string,len,tmp,showValues);
	apendTensor("soma",soma,string,len,tmp,showValues);

	tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, ",\n"PAD"%s\n}", tmp);
	free_mem(tmp);
	return string;
}

char *CamadaSoftMaxing_getGenerate(__attribute__((unused)) CamadaSoftMax self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s()",
			 lname
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
int CamadaSoftMaxing_save(CamadaSoftMax self, FILE *f) {
	if (self->super.erro->error)goto end;
	self->super.erro->addstack(self->super.erro, "CamadaSoftMaxing_save");
	fwrite(&self->super.layer_id, 1, sizeof(int), f);
	fwrite("#", 1, 1, f);
	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}

Camada CamadaSoftMaxing_new(Gpu gpu, Queue queue, P3d size_in, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaSoftMaxing_new");
	CamadaSoftMax self = alloc_mem(1, sizeof(CamadaSoftMax_t));

	P3d size_out = size_in;
	internal_Camada_new((Camada) self, gpu, queue, SOFTMAX_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);

	self->soma = Tensor_new(1, 1, size_in.z, 1, ecx, 0, gpu->context, queue);
	self->exponent = Tensor_new(size_in.x, size_in.z, size_in.z, 1, ecx, 0, gpu->context, queue);


	self->softmaxExp = Kernel_news(gpu->program, "softmaxExp", "Vector entrada, Vector exponent,int k0");
	CheckKernel(softmaxExp);
	self->softmaxSomaExp = Kernel_news(gpu->program, "softmaxSomaExp", "Vector eps, Vector soma, int saidatx, int saidaty, int k0");
	CheckKernel(softmaxSomaExp);
	self->softmaxNormaliza = Kernel_news(gpu->program, "softmaxNormaliza", "Vector exponet, Vector soma, Vector saida,int saidatx, int saidaty, int k0");
	CheckKernel(softmaxNormaliza);
	self->softMaxcalcgrad = Kernel_news(gpu->program, "softMaxcalcgrad", "Vector gradentrada, Vector entrada, Vector gradnext, int k0");
	CheckKernel(softMaxcalcgrad);

	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaSoftMaxing_release;
	self->super.propagation = (int (*)(void *)) CamadaSoftMaxing_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaSoftMaxing_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaSoftMaxing_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaSoftMaxing_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaSoftMaxing_save;
	return (Camada) self;
}
