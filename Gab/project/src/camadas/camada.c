//
// Created by hslhe on 13/11/2021.
//

#include "camadas/camada.h"
#include <math.h>

P3d internnal_getOutSize(Camada self) {
	return (P3d) {self->s->x, self->s->y, self->s->z};
}

void internal_Camada_new(Camada self, Gpu gpu, Queue queue, char layer_id, const char *layer_name, Parametros params, Tensor entrada, P3d dim_in, P3d dim_out, Ecx erro) {
	erro->addstack(erro, "internal_Camada_new");
	self->ecx = erro;
	self->a = entrada;
	self->size_in = dim_in;
	if (entrada) {
		self->da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, erro, 0, gpu->context, queue);
		if (self->ecx->error) {
			goto methods;
		}
	}
	self->s = Tensor_new(dim_out.x, dim_out.y, dim_out.z, 1, erro, 0, gpu->context, queue);
	methods:
	memcpy((void *) &self->layer_id, &layer_id, sizeof(const char));
	memcpy(self, &layer_name, sizeof(const char *));
	self->maxcompute = &gpu->maxworks;
	self->params = params;
	erro->popstack(erro);
	self->getOutSize = (P3d (*)(void *)) internnal_getOutSize;
	self->updateHitLearn = (int (*)(void *, size_t)) internal_updateHitLearn;
	self->queue = queue;
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
	char *string = NULL;
	int len = 0;
	char *tmp = NULL;
	apendstr(string, len, "\"layer_name\":\"%s\",\n\"layer_id\":%d", self->layer_name, self->layer_id);
	if (self->a) {
		tmp = self->a->json(self->a, showValues);
		apendstr(string, len, ",\n"PAD"\"entrada\":%s", tmp);
		gab_free(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"entrada\":null");
	}
	if (self->da) {
		tmp = self->da->json(self->da, showValues);
		apendstr(string, len, ",\n"PAD"\"grad_entrada\":%s", tmp);
		gab_free(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"grad_entrada\":null");
	}
	if (self->s) {
		tmp = self->s->json(self->s, showValues);
		apendstr(string, len, ",\n"PAD"\"saida\":%s", tmp);
		gab_free(tmp);
	} else {
		apendstr(string, len, ",\n"PAD"\"saida\":null");
	}
	apendstr(string, len, ",\n"PAD"\"max_compute\":%zu,\n"
			PAD"\"params\":{\"hitlearn\":%g,\"momento\":%g,\"decaimento\":%g,\"treinavel\":%d}", *self->maxcompute, (double) self->params.hitlearn, (double) self->params.momento, (double) self->params.decaimento, !self->params.skipLearn);

	return string;

}


void internal_saveCamada(FILE *f, Camada self) {
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
	fwrite(&t->flag, sizeof(char), 1, f);
	fwrite(&t->length, sizeof(size_t), 1, f);
	fwrite(&t->bytes, sizeof(size_t), 1, f);
	void *data = t->getvalues(t, NULL);
	fwrite(data, 1, t->bytes, f);
	gab_free(data);
}

void internal_loadTensor(FILE *f, Tensor t, uint32_t size_element) {
	TensorFlag flag;
	size_t length;
	size_t bytes;

	fread(&flag.flag, sizeof(char), 1, f);
	fread(&length, sizeof(size_t), 1, f);
	fread(&bytes, sizeof(size_t), 1, f);
	void *data = gab_alloc(bytes, 1);
	REAL *dtaux;
	fread(data, 1, bytes, f);
	if (!(flag.inteiro || flag.caractere)) {
		if (size_element != sizeof(REAL)) {
			dtaux = gab_alloc(length, sizeof(REAL));
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
	gab_free(data);
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
	return self->retroPropagation(self, ds);
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
	self->params.hitlearn = self->params.lr_0 * pow(self->params.a, iter / self->params.b);
	printf("\n");
	return 0;
}

void internal_Camada_fprint(void *selfp, FILE *destino, char *format, va_list v) {
	vfprintf(destino, format, v);
	va_end(v);
	Camada self = selfp;
	fprintf(destino, "Params %f %f %f %d\n", self->params.hitlearn, self->params.momento, self->params.decaimento, self->params.skipLearn);
	if (self->da) {
		fprintf(destino, "da ->");
		self->da->fprint(self->da, destino);
	}
	if (self->a) {
		fprintf(destino, "a ->");
		self->a->fprint(self->a, destino);
	}
	fprintf(destino, "s ->");
	self->s->fprint(self->s, destino);
}

void internal_compile(Camada self, Gpu gpu) {
	ECXPUSH(self->ecx);
	self->program = clCreateProgramWithSource(gpu->context, 1, (const char **) &self->kernel, &self->kernel_len, self->ecx->perro);
	if (self->ecx->error) {
		char *err = Gpu_errormsg(self->ecx->error);
		fflush(stdout);
		fprintf(stderr, "Error %d  in file %s\n%s\n", self->ecx->error, __FILE__, err);
		self->ecx->pushMsg(self->ecx, "Error %d  in file %s:1\n%s\n", self->ecx->error, __FILE__, err);
		fflush(stderr);
		gab_free(err);

	}
	self->ecx->setError(self->ecx,clBuildProgram(self->program, 1, &gpu->device, NULL, NULL, NULL),"%s:%d %s",__FILE__,__LINE__,__FUNCTION__);
	if (self->ecx->error != CL_SUCCESS) {
		char *buff = NULL;
		size_t len = 0;
		clGetProgramBuildInfo(self->program, gpu->device, CL_PROGRAM_BUILD_LOG, 0, buff, &len);
		buff = gab_alloc(len + 1, 1);
		clGetProgramBuildInfo(self->program, gpu->device, CL_PROGRAM_BUILD_LOG, len, buff, NULL);
		fprintf(stderr, "Error %d  in file %s\n%s",self->ecx->error,__FILE__,  buff);
		clReleaseProgram(self->program);
		gab_free(buff);

	}
//		self->ecx->pushMsg(self->ecx, "Error %d  in file %s at line %d\n%s\n", self->ecx->error, __FILE__, __LINE__, err);
	ECXPOP(self->ecx);
}
