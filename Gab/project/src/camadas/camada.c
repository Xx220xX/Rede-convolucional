//
// Created by hslhe on 13/11/2021.
//

#include "camadas/camada.h"
#include <math.h>

P3d internnal_getOutSize(Camada self) {
	return (P3d) {self->s->x, self->s->y, self->s->z};
}

void internal_Camada_new(Camada self, Gpu gpu, Queue queue, char layer_id, const char *layer_name, Parametros params, Tensor entrada, P3d dim_in, P3d dim_out, Ecx erro) {
	ECX_RETURN_IF_ERROR(self->ecx,)
	self->ecx = erro;
	self->a = entrada;
	self->size_in = dim_in;
	if (entrada) {
		ECX_IF_OK(self->ecx) {
			self->da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, erro, 0, gpu->context, queue);
		}
	}
	ECX_IF_OK(self->ecx) {
		self->s = Tensor_new(dim_out.x, dim_out.y, dim_out.z, 1, erro, 0, gpu->context, queue);
	}
	methods:
	memcpy((void *) &self->layer_id, &layer_id, sizeof(const char));
	memcpy(self, &layer_name, sizeof(const char *));
	self->maxcompute = &gpu->maxworks;
	self->params = params;
	erro->popstack(erro);
	self->getOutSize = (P3d (*)(void *)) internnal_getOutSize;
	self->updateHitLearn = (int (*)(void *, size_t)) internal_updateHitLearn;
	self->queue = queue;
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
}

void internal_Camada_release(Camada *self) {
	if (!self) {
		return;
	}
	if (!*self) {
		return;
	}
	Release((*self)->da);
	Release((*self)->s);
	if ((*self)->program) {
		clReleaseProgram((*self)->program);
	}
	if ((*self)->kernel) {
		gab_free((*self)->kernel);
	}
	(*self)->kernel_len = 0;
}

char *internal_json(Camada self, int showValues) {
	ECX_RETURN_IF_ERROR(self->ecx, NULL)
	char *string = NULL;
	int len = 0;
	char *tmp = NULL;
	apendstr(string, len, "\"layer_name\":\"%s\",\n\"layer_id\":%d", self->layer_name, self->layer_id);
	if (self->a) {
		tmp = self->a->json(self->a, showValues);
		ECX_IF_FAILED(self->ecx, end)
		apendstr(string, len, ",\n"PAD"\"entrada\":%s", tmp);
		gab_free(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"entrada\":null");
	}
	if (self->da) {
		tmp = self->da->json(self->da, showValues);
		ECX_IF_FAILED(self->ecx, end)
		apendstr(string, len, ",\n"PAD"\"grad_entrada\":%s", tmp);
		gab_free(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"grad_entrada\":null");
	}
	if (self->s) {
		tmp = self->s->json(self->s, showValues);
		ECX_IF_FAILED(self->ecx, end)
		apendstr(string, len, ",\n"PAD"\"saida\":%s", tmp);
		gab_free(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"saida\":null");
	}
	apendstr(string, len, ",\n"PAD"\"max_compute\":%zu,\n"
			PAD"\"params\":{\"hitlearn\":%g,\"momento\":%g,\"decaimento\":%g,\"treinavel\":%d}", *self->maxcompute, (double) self->params.hitlearn, (double) self->params.momento, (double) self->params.decaimento, !self->params.skipLearn);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return string;

}


void internal_saveCamada(FILE *f, Camada self) {
	ECX_RETURN_IF_ERROR(self->ecx,)
	fwrite(&self->layer_id, 1, sizeof(char), f);
	fwrite("#", 1, sizeof(char), f);
	uint32_t size_REAL = sizeof(REAL);
	fwrite(&size_REAL, 1, sizeof(uint32_t), f);
	fwrite(&self->params.hitlearn, 1, size_REAL, f);
	fwrite(&self->params.momento, 1, size_REAL, f);
	fwrite(&self->params.decaimento, 1, size_REAL, f);
	fwrite(&self->params.skipLearn, 1, sizeof(int), f);
	fwrite(&self->size_in, 1, sizeof(P3d), f);
}

#define REALCAST(dest, size_element, aux)\
    if((size_element) == sizeof(REAL))\
        (dest) = (aux).auxR;\
    else if((size_element) == sizeof(float)){\
        (dest) = (double)(aux).auxf;\
    }else\
        (dest) = (float)(aux).auxd

void internal_loadCamada(FILE *f, Parametros *parametros, P3d *size_in, uint32_t *size_element) {

	char flag;
	union {
		double auxd;
		float auxf;
		REAL auxR;
	} aux;

	fread(&flag, 1, sizeof(char), f);
	if (flag != '#') {
		fread(&flag, 1, sizeof(char), f);
	}
	fread(size_element, 1, sizeof(uint32_t), f);

	fread(&aux, 1, *size_element, f);
	REALCAST(parametros->hitlearn, *size_element, aux);
	fread(&aux, 1, *size_element, f);
	REALCAST(parametros->momento, *size_element, aux);
	fread(&aux, 1, *size_element, f);
	REALCAST(parametros->decaimento, *size_element, aux);
	fread(&parametros->skipLearn, 1, sizeof(int), f);
	fread(size_in, 1, sizeof(P3d), f);
}

void internal_saveTensor(FILE *f, Tensor t) {
	ECX_RETURN_IF_ERROR(t->ecx,)
	fwrite(&t->flag, sizeof(char), 1, f);
	fwrite(&t->length, sizeof(size_t), 1, f);
	fwrite(&t->bytes, sizeof(size_t), 1, f);
	void *data = t->getvalues(t, NULL);
	fwrite(data, 1, t->bytes, f);
	gab_free(data);
	ECX_REGISTRE_FUNCTION_IF_ERROR(t->ecx)
}

void internal_loadTensor(FILE *f, Tensor t, uint32_t size_element) {
	ECX_RETURN_IF_ERROR(t->ecx,)
	TensorFlag flag;
	size_t length;
	size_t bytes;

	fread(&flag.flag, sizeof(char), 1, f);
	fread(&length, sizeof(size_t), 1, f);
	fread(&bytes, sizeof(size_t), 1, f);
	void *data = gab_alloc(bytes, 1);
	ECX_TRY(t->ecx, !data, end, GAB_FAILED_ALLOC_MEM, "calloc return null ");
	REAL *dtaux;
	fread(data, 1, bytes, f);
	if (!(flag.inteiro || flag.caractere)) {
		if (size_element != sizeof(REAL)) {
			dtaux = gab_alloc(length, sizeof(REAL));
			ECX_TRY(t->ecx, !dtaux, end, GAB_FAILED_ALLOC_MEM, "calloc return null ");
			if (size_element == sizeof(double)) {
				for (int i = 0; i < length; ++i) {
					dtaux[i] = (REAL) ((double *) data)[i];
				}
			} else {
				for (int i = 0; i < length; ++i) {
					dtaux[i] = (REAL) ((float *) data)[i];
				}
			}
			gab_free(data);
			data = dtaux;

		}
	}
	t->setvalues(t, data);
	end:
	gab_free(data);
	ECX_REGISTRE_FUNCTION_IF_ERROR(t->ecx)
}

void internal_saveREAL(FILE *f, REAL value) {
	fwrite(&value, sizeof(REAL), 1, f);
}

void internal_loadREAL(FILE *f, REAL *value, uint32_t size_element) {
	union {
		double auxd;
		float auxf;
		REAL auxR;
	} aux;
	fread(&aux, size_element, 1, f);
	if (size_element == sizeof(REAL)) {
		*value = aux.auxR;
	} else if (size_element != sizeof(double)) {
		*value = (double) aux.auxf;
	} else {
		*value = (float) aux.auxd;
	}
}

int internal_unused(void *a, ...) {
	return 0;
}

int internal_notBatch(Camada self, Tensor ds, size_t batchSize) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	int r = self->retroPropagation(self, ds);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return r;
}

RdParams internal_getDefaultRDP(int is_reluActivation, size_t inputLength, size_t outLength) {
	double a;
	if (is_reluActivation) {
		return RDP(TENSOR_GAUSSIAN, sqrt(2.0 / inputLength), 0);
	}
	a = sqrt(6.0 / (inputLength + outLength));
	return RDP(TENSOR_UNIFORM, 2 * a, -a);
}

int internal_updateHitLearn(Camada self, size_t iter) {
	if (self->params.a == 0) {
		return 0;
	}
	ECX_TRY(self->ecx, self->params.b == 0.0, end, GAB_INVALID_PARAM, "O parametro B não pode ser nulo")
	self->params.hitlearn = self->params.lr_0 * pow(self->params.a, iter / self->params.b);
	end:
	return self->ecx->error;
}

void internal_Camada_fprint(void *selfp, FILE *destino, char *format, va_list v) {
	Camada self = selfp;
	vfprintf(destino, format, v);
	va_end(v);
	ECX_RETURN_IF_ERROR(self->ecx,)
	fprintf(destino, "Params %f %f %f %d\n", self->params.hitlearn, self->params.momento, self->params.decaimento, self->params.skipLearn);
	if (self->da) {
		fprintf(destino, "da ->");
		self->da->fprint(self->da, destino);
		ECX_IF_FAILED(self->ecx, end)
	}
	if (self->a) {
		fprintf(destino, "a ->");
		self->a->fprint(self->a, destino);
		ECX_IF_FAILED(self->ecx, end)
	}
	fprintf(destino, "s ->");
	self->s->fprint(self->s, destino);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return;
}

void internal_compile(Camada self, Gpu gpu) {
	ECX_RETURN_IF_ERROR(self->ecx,)
	self->program = clCreateProgramWithSource(gpu->context, 1, (const char **) &self->kernel, &self->kernel_len, self->ecx->perro);
	if (self->ecx->error) {
		char *err = Gpu_errormsg(self->ecx->error);
		fflush(stdout);
		fprintf(stderr, "Error %d  in file %s\n%s\n", self->ecx->error, __FILE__, err);
		self->ecx->pushMsg(self->ecx, "Error %d  in file %s:1\n%s\n", self->ecx->error, __FILE__, err);
		fflush(stderr);
		gab_free(err);

	}
	self->ecx->setError(self->ecx, clBuildProgram(self->program, 1, &gpu->device, NULL, NULL, NULL), "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
	if (self->ecx->error != CL_SUCCESS) {
		char *buff = NULL;
		size_t len = 0;
		clGetProgramBuildInfo(self->program, gpu->device, CL_PROGRAM_BUILD_LOG, 0, buff, &len);
		buff = gab_alloc(len + 1, 1);
		clGetProgramBuildInfo(self->program, gpu->device, CL_PROGRAM_BUILD_LOG, len, buff, NULL);
		fprintf(stderr, "Error %d  in file %s\n%s", self->ecx->error, __FILE__, buff);
		clReleaseProgram(self->program);
		gab_free(buff);

	}
//	self->ecx->pushMsg(self->ecx, "Error %d  in file %s at line %d\n%s\n", self->ecx->error, __FILE__, __LINE__, err);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
}

void internal_putFativacao(char **s, int *len, FAtivacao_t fAtivacao) {
	FAtivacao fa = {.mask = fAtivacao};
	switch (fa.id) {
		case FSIGMOID: apendstr((*s), (*len), "FSIGMOID");
			break;
		case FTANH: apendstr((*s), (*len), "FTANH");
			break;

		case FLRELU: apendstr((*s), (*len), "FLRELU(%.10g, %.10g)", fa.less, fa.greater);
			break;

		case FLIN: apendstr((*s), (*len), "FLIN");
			break;

		case FALAN: apendstr((*s), (*len), "FALAN");
			break;

		case FSOFTMAX: apendstr((*s), (*len), "FSOFTMAX(%.10g)", fa.epsilon);
	}
}