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
	self->erro->addstack(self->erro, "Tensor_print");
	if (self->erro->error)return NULL;

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
	self->erro->popstack(self->erro);
	return string;
}

char *Tensor_printhex(Tensor self) {
	self->erro->addstack(self->erro, "Tensor_print");
	if (self->erro->error)return NULL;
	size_t id;
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
					id = y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y;
					id = id * self->size_element;
					apendstr(string, len, "0x%02X", (int) m.caractere[id])
					for (int i = 1; i < self->size_element; ++i) {
						apendstr(string, len, "%02X", (int) m.caractere[id + i])
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
	self->erro->popstack(self->erro);
	return string;
}

char *Tensor_json(Tensor self, int showValues) {
	self->erro->addstack(self->erro, "Tensor_json");
	char *string;
	char *tmp = "";
	if (showValues == 2)
		tmp = Tensor_printhex(self);
	else if (showValues)
		tmp = Tensor_print(self);

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
			PAD"\"length\":%d,\n"
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
			 self->lenght,
			 self->bytes,
			 self->size_element,
			 tmp,
			 sizeof(Tensor_t)

	)
	if (showValues)
		free(tmp);
	self->erro->popstack(self->erro);
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

int Tensor_fillM(Tensor self, size_t offset, size_t bytes, void *patern, size_t size_patern);

int Tensor_fill(Tensor self, char partern);

Tensor Tensor_new(size_t x, size_t y, size_t z, size_t w, Ecx ecx, int flag, ...) {
	ecx->addstack(ecx, "Tensor_new");
	Tensor self = calloc(1, sizeof(Tensor_t));
	self->erro = ecx;
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
	self->lenght = self->x * self->y * self->z * self->w;
	self->bytes = self->lenght * self->size_element;

	// alocar memoria
	if (self->flag.ram) {
		self->data = calloc(self->x * self->y * self->z * self->w, self->size_element);
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
			self->data = clCreateBuffer(self->context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, self->bytes, p, &self->erro->error);
		} else {
			self->data = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->bytes, NULL, &self->erro->error);
		}
		va_end(v);

	}

	// metodos
	self->json = Tensor_json;
	self->release = Tensor_release;
	self->registreError = Tensor_registreError;
	self->setvalues = Tensor_setvalues;
	self->getvalues = Tensor_getvalues;
	self->getvaluesM = Tensor_getvaluesm;
	self->setvaluesM = Tensor_setvaluesm;
	self->randomize = Tensor_randomize;
	self->fill = Tensor_fill;
	self->fillM = Tensor_fillM;
	if (self->flag.inteiro)
		self->imagegray = Tensor_imagegrayINT;
	else if (self->flag.caractere)
		self->imagegray = Tensor_imagegrayCHAR;
	else
		self->imagegray = Tensor_imagegrayREAL;
	ecx->popstack(ecx);
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
	self->erro->addstack(self->erro, "Tensor_setvaluesm");
	if (self->erro->error)return self->erro->error;
	if (self->flag.ram) {
		memcpy(self->data + offset, data, bytes);
		return 0;
	} else if (self->flag.shared) {
		fprintf(stderr, "ERROR: shared memory not implanted\n");
	} else {
		self->erro->error = clEnqueueWriteBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	self->erro->popstack(self->erro);
	return self->erro->error;
}

void *Tensor_getvaluesm(Tensor self, size_t offset, void *data, size_t bytes) {
	self->erro->addstack(self->erro, "Tensor_getvaluesm");
	if (self->erro->error)return NULL;
	if (!data)
		data = calloc(bytes, 1);
	if (self->flag.ram) {
		memcpy(data, self->data + offset, bytes);
	} else if (self->flag.shared) {
		fprintf(stderr, "ERROR: shared memory not implanted\n");

	} else {
		self->erro->error = clEnqueueReadBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	self->erro->popstack(self->erro);
	return data;
}


int Tensor_randomize(Tensor self, int type, REAL a, REAL b) {
	if (self->erro->error)return self->erro->error;
	self->erro->addstack(self->erro, "Tensor_randomize");
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
	self->erro->popstack(self->erro);
	return self->erro->error;
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
	if (self->erro->error)return self->erro->error;
	self->erro->addstack(self->erro, "Tensor_imagegrayREAL");
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	REAL *data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	normalizeReal(data, self->x * self->y, 255);
	for (int i = 0; i < h; ++i) {
		if (i + i0 >= height_tensor) {
			self->erro->error = 1;
			goto end;
		}
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= width) {
				self->erro->error = 1;
				goto end;
			}
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	end:
	free(data);
	self->erro->popstack(self->erro);
	return 0;
}

int Tensor_imagegrayINT(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	if (self->erro->error)return self->erro->error;
	self->erro->addstack(self->erro, "Tensor_imagegrayINT");
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	int *data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
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
	self->erro->popstack(self->erro);
	return 0;
}

int Tensor_imagegrayCHAR(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	if (self->erro->error)return self->erro->error;
	self->erro->addstack(self->erro, "Tensor_imagegrayCHAR");
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	ubyte *data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= height_tensor)exit(-1);
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	free(data);
	self->erro->popstack(self->erro);
	return 0;
}

int Tensor_fill(Tensor self, char partern) {
	Tensor_fillM(self, 0, self->bytes, &partern, 1);
}

int Tensor_fillM(Tensor self, size_t offset, size_t bytes, void *patern, size_t size_patern) {
	self->erro->addstack(self->erro, "Tensor_fillM");
	if (!patern || size_patern <= 0) {
		self->erro->error = -1;
		return self->erro->error;
	}
	if (self->flag.ram) {
		void *p = self->data + offset;
		void *pend = self->data + offset + bytes;
		for (; p <= pend; p += size_patern) {
			if (p + size_patern <= pend)
				memcpy(p, patern, size_patern);
		}
	} else if (self->flag.shared) {

	} else {
		self->erro->error = clEnqueueFillBuffer(self->queue, self->data, patern, size_patern, offset, bytes, 0, NULL, NULL);
	}
	self->erro->popstack(self->erro);
}