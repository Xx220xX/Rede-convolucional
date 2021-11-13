//
// Created by hslhe on 11/11/2021.
//

#include "tensor/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <CL/cl.h>
#include "gpu/Gpu.h"



#define asprintf(str, format, ...){int len = snprintf(NULL,0,format,## __VA_ARGS__);\
str = calloc(len+1,1);                                                           \
snprintf(str,len,format,## __VA_ARGS__);}
#define PAD " "

#define apendstr(str, len, format, ...) { \
         size_t sz = snprintf(NULL,0,format,##__VA_ARGS__); \
         if(!str)                         \
         str = calloc(1,sz+1);    \
         else                                 \
         str = realloc(str,len+sz+1);                              \
         char *tmp = str+len;               \
         len = len+sz;\
         sprintf(tmp,format,##__VA_ARGS__) ;                           \
         \
         \
}\


char *Tensor_print(Tensor self) {
	if (self->error)return NULL;
	Memory m = {0};
	m.mem = self->getvalues(self, NULL);
	size_t len = 0;
	char *string = NULL;
	for (int w = 0; w < self->w; ++w) {
		if (self->flag.dimensao4D) {
			if (w == 0) apendstr(string, len, "[")
			else apendstr(string, len, "\n,[")
		}

		for (int z = 0; z < self->z; ++z) {
			if (z == 0) apendstr(string, len, "[")
			else apendstr(string, len, ",[")

			for (int x = 0; x < self->x; ++x) {
				if (x == 0) apendstr(string, len, "[")
				else apendstr(string, len, ",[")
				for (int y = 0; y < self->y; ++y) {
					if (y > 0) apendstr(string, len, ", ");
					if (self->flag.inteiro) {
						apendstr(string, len, "%d", m.inteiro[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					} else if (self->flag.caractere) {
						apendstr(string, len, "0x%02X", (int) m.caractere[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					} else {
						apendstr(string, len, "%.3lf", (double) m.real[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					}
				}
				apendstr(string, len, "]")
			}
			apendstr(string, len, "]")
		}

		if (self->flag.dimensao4D) {
			apendstr(string, len, "]")
		}
	}
	free(m.mem);
	return string;
}

char *Tensor_json(Tensor self) {
	char *string;
	char *tmp = Tensor_print(self);
	asprintf(string, "{\n"
			PAD"\"flag\":{\n"
			PAD PAD "\"dimensao4D\":%d,\n"
			PAD PAD"\"ram\":%d,\n"
			PAD PAD"\"caractere\":%d,\n"
			PAD PAD"\"inteiro\":%d,\n"
			PAD PAD"\"shared\":%d,\n"
			PAD PAD"\"size\":%llu\n"
			PAD"},\n"
			PAD"\"flagn\":%d,\n"
			PAD"\"x\":%d,\n"
			PAD"\"y\":%d,\n"
			PAD"\"z\":%d,\n"
			PAD"\"w\":%d,\n"
			PAD"\"bytes\":%d,\n"
			PAD"\"size_element\":%d,\n"
			PAD"\"data\":[%s],\n"
			PAD"\"size\":%d\n"

			   "}",
			 (int) self->flag.dimensao4D,
			 (int) self->flag.ram,
			 (int) self->flag.caractere,
			 (int) self->flag.inteiro,
			 (int) self->flag.shared,
			 sizeof(TensorFlag),
			 (int) self->flag.flag,
			 self->x,
			 self->y,
			 self->z,
			 self->w,
			 self->bytes,
			 self->size_element,
			 tmp,
			 sizeof(Tensor_t)

	)
	free(tmp);
	return string;
}

int Tensor_imagegrayREAL(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l);

int Tensor_imagegrayINT(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l);

int Tensor_imagegrayCHAR(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l);

void *Tensor_getvalues(Tensor self, void *data);

int Tensor_setvalues(Tensor self, void *data);

void *Tensor_getvaluesm(Tensor self, size_t offset, void *data, size_t bytes);

int Tensor_setvaluesm(Tensor self, size_t offset, void *data, size_t bytes);

int Tensor_randomize(Tensor self, int type, REAL a, REAL b);

void Tensor_registreError(Tensor self, char *format, ...);

void Tensor_release(Tensor *self);

Tensor Tensor_new(size_t x, size_t y, size_t z, size_t w, int flag, ...) {
	Tensor self = calloc(1, sizeof(Tensor_t));
	memset(self, flag, 1);
	// verificando flag
	if (self->flag.inteiro && self->flag.caractere) {
		Tensor_registreError(self, "flag invalida: inteiro e caractere = 1");
		free(self);
		return NULL;
	}
	if (self->flag.ram && self->flag.shared) {
		Tensor_registreError(self, "flag invalida: ram e shared = 1");
		free(self);
		return NULL;
	}
	self->x = x;
	self->y = y;
	self->z = z;
	self->w = w;
	if (!self->flag.dimensao4D)
		self->w = 1;
	self->size_element = sizeof(REAL);
	if (self->flag.inteiro)
		self->size_element = sizeof(int);
	else if (self->flag.caractere)
		self->size_element = sizeof(char);
	self->bytes = self->x * self->y * self->z * self->w * self->size_element;

	// alocar memoria
	if (self->flag.ram) {
		self->data = calloc(self->x*self->y*self->z*self->w, self->size_element);
	} else if (self->flag.shared) {
		fprintf(stderr, "Invalid flag: Shared memory not suported");
		exit(-1);
	} else {
		va_list v;
		va_start(v, flag);
		self->context = va_arg(v, void *);
		self->queue = va_arg(v, void *);
		void *p = NULL;

		if (self->flag.copy) {
			p = va_arg(v, void *);
			self->data = clCreateBuffer(self->context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, self->bytes, p, &self->error);
		} else {
			self->data = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->bytes, NULL, &self->error);

		}
		va_end(v);

	}

	// metodos
	self->json = (char *(*)(void *)) Tensor_json;
	self->release = (void (*)(void *)) Tensor_release;
	self->registreError = (void (*)(void *, void *, ...)) Tensor_registreError;
	self->setvalues = (int (*)(void *, void *)) Tensor_setvalues;
	self->getvalues = (void *(*)(void *, void *)) Tensor_getvalues;
	self->getvaluesM = (void *(*)(void *, size_t, void *, size_t)) Tensor_getvaluesm;
	self->setvaluesM = (int (*)(void *, size_t, void *, size_t)) Tensor_setvaluesm;
	self->randomize = (int (*)(void *, int, REAL, REAL)) Tensor_randomize;
	if (self->flag.inteiro)
		self->imagegray = (int (*)(void *, ubyte *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t)) Tensor_imagegrayINT;
	else if (self->flag.caractere)
		self->imagegray = (int (*)(void *, ubyte *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t)) Tensor_imagegrayCHAR;
	else
		self->imagegray = (int (*)(void *, ubyte *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t)) Tensor_imagegrayREAL;

	return self;
}

void Tensor_release(Tensor *self) {
	if (!self)return;
	if (!*self)return;
	if ((*self)->data) {
		if ((*self)->flag.ram) {
			free((*self)->data);
		} else if ((*self)->flag.shared) {

		} else {
			clReleaseMemObject((*self)->data);
		}
	}
	free(self[0]);
	*self = NULL;
}

void Tensor_registreError(Tensor self, char *format, ...) {
	va_list v;
	va_start(v, format);
	FILE *f;
	if (self->file_debug)
		f = fopen(self->file_debug, "a");
	else
		f = stderr;// fopen("tensor.debug", "a");
	vfprintf(f, format, v);
	va_end(v);
	if (self->file_debug)
		fclose(f);

}

int Tensor_setvalues(Tensor self, void *data) {
	return self->setvaluesM(self, 0, data, self->bytes);
}

void *Tensor_getvalues(Tensor self, void *data) {
	return self->getvaluesM(self, 0, data, self->bytes);
}


int Tensor_setvaluesm(Tensor self, size_t offset, void *data, size_t bytes) {
	if (self->error)return self->error;
	if (self->flag.ram) {
		memcpy(self->data + offset, data, bytes);
		return 0;
	} else if (self->flag.shared) {
		fprintf(stderr, "ERROR: shared memory not implanted\n");
	} else {
		self->error = clEnqueueWriteBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	return self->error;
}

void *Tensor_getvaluesm(Tensor self, size_t offset, void *data, size_t bytes) {
	if (self->error)return NULL;
	if (!data)
		data = calloc(bytes, 1);
	if (self->flag.ram) {
		memcpy(data, self->data + offset, bytes);
	} else if (self->flag.shared) {
		fprintf(stderr, "ERROR: shared memory not implanted\n");

	} else {
		self->error = clEnqueueReadBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	return data;
}


int Tensor_randomize(Tensor self, int type, REAL a, REAL b) {
	if (self->error)return self->error;
	void *m = calloc(self->bytes, 1);
	size_t len = self->bytes / self->size_element;
	REAL x;
	for (int i = 0; i < len; ++i) {
		if (type == TENSOR_NORMAL) {
			x = Tensor_randn() * a + b;
		} else {
			x = Tensor_rand() * a + b;
		}
		if (self->flag.caractere)
			((char *) m)[i] = (char) x;
		else if (self->flag.inteiro)
			((int *) m)[i] = (int) x;
		else
			((REAL *) m)[i] = (REAL) x;
	}
	self->setvalues(self, m);
	free(m);
	return self->error;
}

void normalizeReal(REAL *data, size_t len, REAL a) {

	REAL mx = data[0], mn = data[0];
	for (int i = 1; i < len; ++i) {
		if (mx < data[i]) mx = data[i];
		if (mn > data[i]) mn = data[i];
	}
	mx = mx - mn;
	a = a / mx;
	for (int i = 0; i < len; ++i) {
		data[i] = (data[i] - mn) * a;
	}
}

void normalizeInt(int *data, size_t len, REAL a) {

	int mx = data[0], mn = data[0];
	for (int i = 1; i < len; ++i) {
		if (mx < data[i]) mx = data[i];
		if (mn > data[i]) mn = data[i];
	}
	mx = mx - mn;
	a = a / (REAL) mx;
	for (int i = 0; i < len; ++i) {
		data[i] = (int) ((REAL) (data[i] - mn) * a);
	}
}


int Tensor_imagegrayREAL(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	if (self->error)return self->error;
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	REAL *data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y)*self->size_element, NULL, self->size_element * self->x * self->y);
	normalizeReal(data, self->x * self->y, 255);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= height_tensor)exit(-1);
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	free(data);
	return 0;
}

int Tensor_imagegrayINT(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	if (self->error)return self->error;
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	int *data = self->getvaluesM(self,  (z * self->x * self->y + l * self->z * self->x * self->y)*self->size_element, NULL, self->size_element * self->x * self->y);
	normalizeInt(data, self->x * self->y, 255);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= height_tensor)exit(-1);
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y] & 0xff);
		}
	}
	free(data);
	return 0;
}

int Tensor_imagegrayCHAR(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	if (self->error)return self->error;
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	ubyte *data = self->getvaluesM(self,  (z * self->x * self->y + l * self->z * self->x * self->y)*self->size_element, NULL, self->size_element * self->x * self->y);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= height_tensor)exit(-1);
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	free(data);
	return 0;
}
