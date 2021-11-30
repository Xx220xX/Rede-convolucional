//
// Created by Henrique on 19/11/2021.
//
/***
 * Camada Padding adiciona 0 nas laterias da imagem
 *
 */
#include "camadas/CamadaPadding.h"

static const char *lname = "Padding";

void CamadaPadding_release(CamadaPadding *self) {
	Release((*self)->paddingBack);
	Release((*self)->paddingfeed);
	internal_Camada_release((Camada *) self);
	free_mem(*self);
}

int CamadaPadding_propagation(CamadaPadding self) {
	Execute(paddingfeed, self->super.a->length,
			&self->super.a->data, &self->super.s->data,
			&self->super.a->x, &self->super.a->y,
			&self->super.s->x, &self->super.s->y,
			&self->top, &self->left
	);
	return self->super.erro->error;
}

int CamadaPadding_backpropagation(CamadaPadding self, Tensor ds) {
	if (self->super.da) {
		Execute(paddingBack, self->super.a->length,
				&ds->data,
				&self->super.da->data,
				&self->super.a->x,
				&self->super.a->y,
				&self->super.s->x,
				&self->super.s->y,
				&self->top,
				&self->left);
	}

	return self->super.erro->error;

}

char *CamadaPadding_json(CamadaPadding self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp=internal_json((Camada) self, showValues);
	apendstr(string, len,
			 "{"
					 PAD"%s,\n"
					 PAD"\"top\":%u,\n"
					 PAD"\"bottom\":%u,\n"
					 PAD"\"left\":%u,\n"
					 PAD"\"right\":%u"
						"\n}",
			 tmp,
			 self->top,
			 self->bottom,
			 self->left,
			 self->right);
	free_mem(tmp);


	return string;
}

char *CamadaPadding_getGenerate(CamadaPadding self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len,
			 "%s(%u,%u,%u,%u)",
			 lname,
			 self->top,
			 self->bottom,
			 self->left,
			 self->right
	);

	return string;
}

/**
 * Salva a camada conv em um arquivo
 * 4 bytes -> indicando o tipo
 * 1 bytes -> separação deve ser '#'
 * 4 bytes -> top
 * 4 bytes -> bottom
 * 4 bytes -> left
 * 4 bytes -> right
 *
 * @param self camada
 * @param f arquivo para salvar
 * @return 0 caso não detecte nenhuma falha
 */
int CamadaPadding_save(CamadaPadding self, FILE *f) {
	self->super.erro->addstack(self->super.erro, "CamadaPadding_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->top, 1, sizeof(uint32_t), f);
	fwrite(&self->bottom, 1, sizeof(uint32_t), f);
	fwrite(&self->left, 1, sizeof(uint32_t), f);
	fwrite(&self->right, 1, sizeof(uint32_t), f);
	end:
	self->super.erro->popstack(self->super.erro);
	return self->super.erro->error;
}
Camada CamadaPadding_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ecx->addstack(ecx, "CamadaPadding_load");
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;

	uint32_t top,bottom,left,right;

	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&top, sizeof(uint32_t), 1, f);
	fread(&bottom, sizeof(uint32_t), 1, f);
	fread(&left, sizeof(uint32_t), 1, f);
	fread(&right, sizeof(uint32_t), 1, f);

	CamadaPadding self = (CamadaPadding) CamadaPadding_new(gpu, queue, size_in, top,bottom,left,right,entrada,ecx);

	end:
	ecx->popstack(ecx);
	return (Camada) self;
}

Camada CamadaPadding_new(Gpu gpu, Queue queue, P3d size_in, uint32_t top, uint32_t bottom,
						 uint32_t left, uint32_t right, Tensor entrada, Ecx ecx) {

	ecx->addstack(ecx, "CamadaPadding_new");
	CamadaPadding self = alloc_mem(1, sizeof(CamadaPadding_t));
	P3d size_out = {size_in.x + top + bottom, size_in.y + left + right, size_in.z};
	internal_Camada_new((Camada) self, gpu, queue, PADDING_ID, lname, (Parametros) {0}, entrada, size_in, size_out, ecx);

	self->top = top;
	self->bottom = bottom;
	self->left = left;
	self->right = right;

	self->paddingfeed = Kernel_news(gpu->program, "paddingfeed", "Vector in,Vector out,\n"
																 "unsigned int txi,unsigned int tyi,\n"
																 "unsigned int txo,unsigned int tyo,\n"
																 "unsigned int t, unsigned int l ,\n"
																 "int k0");

	CheckKernel(paddingfeed);
	self->paddingBack = Kernel_news(gpu->program, "paddingBack", "Vector gradNext,Vector gradin,\n"
																 "unsigned int txi, unsigned int tyi,\n"
																 "unsigned int txo,unsigned int tyo,\n"
																 "unsigned int t, unsigned int l , int k0");
	CheckKernel(paddingBack);

	ecx->popstack(ecx);
	methods:
	self->super.release = (void (*)(void *)) CamadaPadding_release;
	self->super.propagation = (int (*)(void *)) CamadaPadding_propagation;
	self->super.retroPropagation = (int (*)(void *, Tensor)) CamadaPadding_backpropagation;
	self->super.json = (char *(*)(void *, int)) CamadaPadding_json;
	self->super.getGenerate = (char *(*)(void *)) CamadaPadding_getGenerate;
	self->super.save = (int (*)(void *, FILE *)) CamadaPadding_save;
	return (Camada) self;
}
